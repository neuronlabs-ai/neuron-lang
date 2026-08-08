# ═══════════════════════════════════════════════════════════════════════
#  Python → NEURON (.nr) Transpiler
#  Converts tensor-subset Python (numpy/torch) to NEURON source code
# ═══════════════════════════════════════════════════════════════════════

import ast
import sys
import os
import textwrap

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class NeuronTranspiler(ast.NodeVisitor):
    """
    Transpiles a subset of Python AST into NEURON .nr source code.

    Supported Python constructs:
      - Functions (def → fn)
      - Classes with __init__ + forward/methods (class → model)
      - Variable assignments (x = ... → let x = ...)
      - Binary ops: +, -, *, /, @, %, ==, !=, <, >, <=, >=
      - Unary ops: -, not
      - Function calls: np.matmul, torch.softmax, np.zeros, etc.
      - Control flow: if/else, while, for ... in range(...)
      - Return statements
      - Print statements
      - List literals
      - Attribute access (self.w1)
      - Indexing (x[0])
    """

    # Maps Python/numpy/torch function names → NEURON builtins
    FUNC_MAP = {
        # Tensor creation
        'np.zeros': 'zeros',
        'numpy.zeros': 'zeros',
        'torch.zeros': 'zeros',
        'np.ones': 'zeros',  # ones(m,n) → zeros(m,n) + 1.0
        'numpy.ones': 'zeros',
        'torch.ones': 'zeros',
        'np.random.randn': 'glorot',
        'torch.randn': 'glorot',
        'torch.nn.init.xavier_uniform_': 'glorot',

        # Activations
        'torch.relu': 'relu',
        'torch.nn.functional.relu': 'relu',
        'F.relu': 'relu',
        'torch.gelu': 'gelu',
        'torch.nn.functional.gelu': 'gelu',
        'F.gelu': 'gelu',
        'torch.sigmoid': 'sigmoid',
        'torch.nn.functional.sigmoid': 'sigmoid',
        'F.sigmoid': 'sigmoid',
        'torch.softmax': 'softmax',
        'torch.nn.functional.softmax': 'softmax',
        'F.softmax': 'softmax',
        'np.exp': 'exp',
        'torch.exp': 'exp',
        'np.log': 'log',
        'torch.log': 'log',
        'np.sqrt': 'sqrt',
        'torch.sqrt': 'sqrt',
        'np.mean': 'mean',
        'numpy.mean': 'mean',
        'torch.mean': 'mean',
        'np.abs': 'abs',
        'torch.abs': 'abs',
        'np.tanh': 'tanh',
        'torch.tanh': 'tanh',

        # Loss functions
        'torch.nn.functional.mse_loss': 'mse',
        'F.mse_loss': 'mse',
        'torch.nn.functional.cross_entropy': 'cross_entropy',
        'F.cross_entropy': 'cross_entropy',

        # Tensor ops
        'np.matmul': '@',
        'torch.matmul': '@',
        'np.transpose': 'transpose',
        'torch.transpose': 'transpose',
        'np.concatenate': 'concat',
        'torch.cat': 'concat',
        'torch.concat': 'concat',

        # I/O
        'np.load': 'load_tensor',
        'torch.load': 'load_tensor',
        'np.save': 'save_tensor',
        'torch.save': 'save_tensor',
        'print': 'print',
    }

    # NEURON reserved keywords that might appear as Python VARIABLE NAMES
    # (statement-level keywords like if/while/for/return are not included
    #  because Python also reserves them — they can't be variable names)
    RESERVED_VARS = {'model', 'fn', 'let', 'import', 'temporal', 'causal',
                     'effect', 'mut', 'pub', 'struct', 'enum', 'match',
                     'type', 'trait', 'impl'}

    def __init__(self):
        self.indent_level = 0
        self.output_lines = []
        self.models = {}          # class_name → {fields, methods}
        self.current_class = None
        self.is_init = False

    def indent(self):
        return '  ' * self.indent_level

    def emit(self, line):
        self.output_lines.append(self.indent() + line)

    def emit_raw(self, line):
        self.output_lines.append(line)

    # ─── Top-level dispatch ────────────────────────

    def transpile(self, source_code, source_file="<input>"):
        """Main entry point: parse Python source and emit .nr code."""
        tree = ast.parse(source_code)
        self.emit_raw(f"// Auto-transpiled from {source_file} by NEURON Python Transpiler")
        self.emit_raw(f"// Source: {source_file}")
        self.emit_raw("")

        # First pass: collect class definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._collect_class(node)

        # Second pass: emit models and free functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._emit_model(node)
            elif isinstance(node, ast.FunctionDef):
                self._emit_function(node)
            elif isinstance(node, ast.If):
                # Skip if __name__ == '__main__' guard, emit body as main
                if self._is_main_guard(node):
                    self._emit_main_block(node)
                else:
                    self._emit_if(node)
            elif isinstance(node, ast.Assign):
                self._emit_assign(node, top_level=True)
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    self.emit(self._emit_call_expr(node.value))

        return '\n'.join(self.output_lines)

    # ─── Class → Model ────────────────────────

    def _collect_class(self, node):
        """Pre-collect class fields from __init__."""
        fields = []
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == '__init__':
                    fields = self._extract_init_fields(item)
                else:
                    methods.append(item)
        self.models[node.name] = {'fields': fields, 'methods': methods}

    def _extract_init_fields(self, init_fn):
        """Extract self.xyz = ... assignments from __init__."""
        fields = []
        for stmt in init_fn.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        field_name = target.attr
                        init_expr = self._expr_to_nr(stmt.value)
                        # Infer type from initializer
                        type_str = self._infer_type(stmt.value)
                        fields.append((field_name, type_str, init_expr))
        return fields

    def _infer_type(self, node):
        """Infer NEURON type annotation from a Python expression."""
        if isinstance(node, ast.Call):
            func_name = self._get_full_func_name(node)
            if 'zeros' in func_name or 'ones' in func_name or 'randn' in func_name or 'glorot' in func_name:
                args = [self._expr_to_nr(a) for a in node.args]
                if len(args) == 2:
                    return f"Tensor[{args[0]}, {args[1]}]"
                elif len(args) == 1:
                    return f"Tensor[{args[0]}]"
            if 'Linear' in func_name:
                args = [self._expr_to_nr(a) for a in node.args]
                if len(args) >= 2:
                    return f"Tensor[{args[0]}, {args[1]}]"
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return "Float"
            elif isinstance(node.value, int):
                return "Int"
        return "Tensor"

    def _emit_model(self, node):
        """Emit a Python class as a NEURON model declaration."""
        self.current_class = node.name
        info = self.models[node.name]

        # Collect constructor params (from class __init__ args, excluding self)
        init_fn = None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                init_fn = item
                break

        params = []
        if init_fn:
            for arg in init_fn.args.args:
                if arg.arg != 'self':
                    params.append(arg.arg)

        param_str = ""
        if params:
            param_str = "(" + ", ".join(f"{p}: Int" for p in params) + ")"

        self.emit(f"model {node.name}{param_str}:")
        self.indent_level += 1

        # Emit fields
        for fname, ftype, finit in info['fields']:
            self.emit(f"{fname}: {ftype} = {finit}")

        self.emit_raw("")

        # Emit methods
        for method in info['methods']:
            self._emit_method(method)

        self.indent_level -= 1
        self.emit_raw("")
        self.current_class = None

    def _emit_method(self, node):
        """Emit a Python method as a NEURON fn inside a model."""
        params = []
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
            type_str = self._get_arg_type(arg)
            params.append(f"{arg.arg}: {type_str}")

        ret_type = self._get_return_type(node)
        effect = ""
        # Check if method mutates self
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
                        effect = " [Effect[Mut[self]]]"
                        break

        param_str = ", ".join(params)
        ret_str = f" -> {ret_type}" if ret_type else ""

        self.emit(f"fn {node.name}(self, {param_str}){ret_str}{effect}:")
        self.indent_level += 1
        self._emit_body(node.body)
        self.indent_level -= 1
        self.emit_raw("")

    # ─── Free functions ────────────────────────

    def _emit_function(self, node):
        """Emit a Python function as a NEURON fn."""
        params = []
        for arg in node.args.args:
            type_str = self._get_arg_type(arg)
            name = arg.arg
            if name in self.RESERVED_VARS:
                name = name + '_'
            params.append(f"{name}: {type_str}")

        ret_type = self._get_return_type(node)
        param_str = ", ".join(params)
        ret_str = f" -> {ret_type}" if ret_type else ""

        self.emit(f"fn {node.name}({param_str}){ret_str}:")
        self.indent_level += 1
        self._emit_body(node.body)
        self.indent_level -= 1
        self.emit_raw("")

    # ─── Statements ────────────────────────

    def _emit_body(self, stmts):
        """Emit a list of statements."""
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                self._emit_assign(stmt)
            elif isinstance(stmt, ast.AugAssign):
                self._emit_aug_assign(stmt)
            elif isinstance(stmt, ast.Return):
                self._emit_return(stmt)
            elif isinstance(stmt, ast.If):
                self._emit_if(stmt)
            elif isinstance(stmt, ast.While):
                self._emit_while(stmt)
            elif isinstance(stmt, ast.For):
                self._emit_for(stmt)
            elif isinstance(stmt, ast.Expr):
                expr_str = self._expr_to_nr(stmt.value)
                self.emit(expr_str)
            elif isinstance(stmt, ast.Pass):
                pass  # skip
            elif isinstance(stmt, ast.AnnAssign):
                self._emit_ann_assign(stmt)

    def _emit_assign(self, node, top_level=False):
        """Emit assignment: x = expr → let x = expr"""
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                # self.x = expr → skip in methods (already declared as field)
                return
            name = self._expr_to_nr(target)
            value = self._expr_to_nr(node.value)
            # Handle ones() → zeros() + 1.0
            func_name = ""
            if isinstance(node.value, ast.Call):
                func_name = self._get_full_func_name(node.value)
            if 'ones' in func_name:
                self.emit(f"let {name} = {value} + 1.0")
            else:
                self.emit(f"let {name} = {value}")

    def _emit_ann_assign(self, node):
        """Emit annotated assignment: x: Type = expr"""
        name = self._expr_to_nr(node.target)
        value = self._expr_to_nr(node.value) if node.value else "0"
        self.emit(f"let {name} = {value}")

    def _emit_aug_assign(self, node):
        """Emit augmented assignment: x += expr → let x = x + expr"""
        target = self._expr_to_nr(node.target)
        value = self._expr_to_nr(node.value)
        op = self._binop_to_str(node.op)
        self.emit(f"let {target} = {target} {op} {value}")

    def _emit_return(self, node):
        if node.value:
            self.emit(f"return {self._expr_to_nr(node.value)}")
        else:
            self.emit("return")

    def _emit_if(self, node):
        cond = self._expr_to_nr(node.test)
        self.emit(f"if {cond}:")
        self.indent_level += 1
        self._emit_body(node.body)
        self.indent_level -= 1
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # elif
                elif_node = node.orelse[0]
                cond2 = self._expr_to_nr(elif_node.test)
                self.emit(f"else if {cond2}:")
                self.indent_level += 1
                self._emit_body(elif_node.body)
                self.indent_level -= 1
                if elif_node.orelse:
                    self.emit("else:")
                    self.indent_level += 1
                    self._emit_body(elif_node.orelse)
                    self.indent_level -= 1
            else:
                self.emit("else:")
                self.indent_level += 1
                self._emit_body(node.orelse)
                self.indent_level -= 1

    def _emit_while(self, node):
        cond = self._expr_to_nr(node.test)
        self.emit(f"while {cond}:")
        self.indent_level += 1
        self._emit_body(node.body)
        self.indent_level -= 1

    def _emit_for(self, node):
        var = self._expr_to_nr(node.target)
        iter_expr = self._expr_to_nr(node.iter)
        self.emit(f"for {var} in {iter_expr}:")
        self.indent_level += 1
        self._emit_body(node.body)
        self.indent_level -= 1

    def _emit_main_block(self, node):
        """Emit if __name__ == '__main__' block as fn main()."""
        self.emit("fn main():")
        self.indent_level += 1
        self._emit_body(node.body)
        self.indent_level -= 1
        self.emit_raw("")

    # ─── Expressions ────────────────────────

    def _expr_to_nr(self, node):
        """Convert a Python expression AST node to NEURON source string."""
        if node is None:
            return ""

        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            elif isinstance(node.value, bool):
                return "true" if node.value else "false"
            elif isinstance(node.value, float):
                return str(node.value)
            elif isinstance(node.value, int):
                return str(node.value)
            return str(node.value)

        if isinstance(node, ast.Name):
            name = node.id
            if name in self.RESERVED_VARS:
                name = name + '_'
            return name

        if isinstance(node, ast.Attribute):
            value = self._expr_to_nr(node.value)
            return f"{value}.{node.attr}"

        if isinstance(node, ast.BinOp):
            left = self._expr_to_nr(node.left)
            right = self._expr_to_nr(node.right)
            op = self._binop_to_str(node.op)
            return f"{left} {op} {right}"

        if isinstance(node, ast.UnaryOp):
            operand = self._expr_to_nr(node.operand)
            if isinstance(node.op, ast.USub):
                return f"-{operand}"
            elif isinstance(node.op, ast.Not):
                return f"not {operand}"
            return operand

        if isinstance(node, ast.Compare):
            left = self._expr_to_nr(node.left)
            parts = [left]
            for op, comp in zip(node.ops, node.comparators):
                parts.append(self._cmpop_to_str(op))
                parts.append(self._expr_to_nr(comp))
            return ' '.join(parts)

        if isinstance(node, ast.BoolOp):
            op = " && " if isinstance(node.op, ast.And) else " || "
            return op.join(self._expr_to_nr(v) for v in node.values)

        if isinstance(node, ast.Call):
            return self._emit_call_expr(node)

        if isinstance(node, ast.Subscript):
            value = self._expr_to_nr(node.value)
            slice_val = self._expr_to_nr(node.slice)
            return f"{value}[{slice_val}]"

        if isinstance(node, ast.List):
            elts = ', '.join(self._expr_to_nr(e) for e in node.elts)
            return f"[{elts}]"

        if isinstance(node, ast.Tuple):
            elts = ', '.join(self._expr_to_nr(e) for e in node.elts)
            return f"({elts})"

        if isinstance(node, ast.IfExp):
            # Ternary: a if cond else b
            body = self._expr_to_nr(node.body)
            test = self._expr_to_nr(node.test)
            orelse = self._expr_to_nr(node.orelse)
            return f"if {test} then {body} else {orelse}"

        if isinstance(node, ast.Slice):
            lower = self._expr_to_nr(node.lower) if node.lower else ""
            upper = self._expr_to_nr(node.upper) if node.upper else ""
            return f"{lower}:{upper}"

        return f"/* UNSUPPORTED: {ast.dump(node)} */"

    def _emit_call_expr(self, node):
        """Convert a function call to NEURON syntax."""
        func_name = self._get_full_func_name(node)
        args = [self._expr_to_nr(a) for a in node.args]

        # Check if this maps to a NEURON builtin
        nr_func = self.FUNC_MAP.get(func_name, None)

        if nr_func:
            # For NEURON builtins, strip keyword arguments (e.g., dim=, axis=)
            # NEURON builtins use positional args only
            pass  # Don't add kwargs for builtins
        else:
            # For non-builtin calls, keep keyword arguments
            for kw in node.keywords:
                if kw.arg:
                    args.append(f"{kw.arg}={self._expr_to_nr(kw.value)}")

        # Check if this is a matmul call → use @ operator
        if nr_func == '@':
            if len(args) >= 2:
                return f"{args[0]} @ {args[1]}"

        # Map to NEURON builtin
        if nr_func:
            if nr_func == 'transpose' and len(args) == 1:
                return f"transpose({args[0]}, 0, 1)"
            return f"{nr_func}({', '.join(args)})"

        # Method calls: obj.method(args) → obj.method(args)
        if isinstance(node.func, ast.Attribute):
            obj = self._expr_to_nr(node.func.value)
            method = node.func.attr

            # Handle .T transpose
            if method == 'T' or method == 't':
                return f"transpose({obj}, 0, 1)"

            # Handle .reshape, .view → keep as-is for now
            return f"{obj}.{method}({', '.join(args)})"

        # Direct function call
        return f"{func_name}({', '.join(args)})"

    # ─── Helpers ────────────────────────

    def _get_full_func_name(self, node):
        """Get the full dotted function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return ""

    def _binop_to_str(self, op):
        """Convert Python binary operator to NEURON string."""
        op_map = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.Mod: '%', ast.MatMult: '@', ast.Pow: '**',
            ast.FloorDiv: '/',
        }
        return op_map.get(type(op), '+')

    def _cmpop_to_str(self, op):
        """Convert Python comparison operator to NEURON string."""
        op_map = {
            ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.Gt: '>',
            ast.LtE: '<=', ast.GtE: '>=',
        }
        return op_map.get(type(op), '==')

    def _get_arg_type(self, arg):
        """Extract type annotation from a function argument."""
        if arg.annotation:
            return self._type_ann_to_nr(arg.annotation)
        # If the arg name matches a known model class, use that type
        if arg.arg in self.models:
            return arg.arg
        # Heuristic: param named 'model', 'net', 'network', 'classifier'
        # likely refers to a model class — use the most recently defined one
        model_param_names = {'model', 'net', 'network', 'classifier', 'encoder',
                             'decoder', 'backbone', 'module'}
        if arg.arg in model_param_names and self.models:
            # Use the last defined model class
            last_model = list(self.models.keys())[-1]
            return last_model
        # Detect common scalar parameter names
        scalar_names = {'lr', 'learning_rate', 'alpha', 'beta', 'gamma', 'epsilon',
                        'eps', 'momentum', 'weight_decay', 'dropout', 'rate', 'ratio',
                        'scale', 'temperature', 'tau', 'lambda_', 'threshold'}
        if arg.arg in scalar_names:
            return 'Float'
        return "Tensor"

    def _type_ann_to_nr(self, node):
        """Convert a Python type annotation to NEURON type string."""
        if isinstance(node, ast.Name):
            type_map = {'float': 'Float', 'int': 'Int', 'bool': 'Bool', 'str': 'String'}
            return type_map.get(node.id, node.id)
        if isinstance(node, ast.Subscript):
            base = self._expr_to_nr(node.value)
            if base == 'Tensor' or base == 'torch.Tensor':
                return f"Tensor"
        if isinstance(node, ast.Constant):
            return str(node.value)
        return "Tensor"

    def _get_return_type(self, node):
        """Infer return type from function return annotation or body."""
        if node.returns:
            return self._type_ann_to_nr(node.returns)
        # Walk body for return statements
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                return "Tensor"
        return None

    def _is_main_guard(self, node):
        """Check if this is an `if __name__ == '__main__':` block."""
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python py2nr.py <input.py> [output.nr]")
        print("  Transpiles Python tensor code to NEURON .nr source")
        sys.exit(1)

    input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = os.path.splitext(input_file)[0] + '.nr'

    with open(input_file, 'r', encoding='utf-8') as f:
        source = f.read()

    transpiler = NeuronTranspiler()
    nr_code = transpiler.transpile(source, source_file=input_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(nr_code)

    print(f"[OK] Transpiled {input_file} -> {output_file}")
    print(f"     Lines: {len(nr_code.splitlines())}")
    print(f"     Size:  {len(nr_code.encode('utf-8'))} bytes")
    print("")
    print("--- Generated NEURON Code ---")
    print(nr_code)


if __name__ == '__main__':
    main()
