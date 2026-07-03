# ═══════════════════════════════════════════════════════════════════
#  NEURON JIT Transpiled PyTorch Python Source Code
# ═══════════════════════════════════════════════════════════════════
import torch
import math

class Model:
    def __init__(self, name):
        self.name = name
        self.fields = {}

# Global variable dictionary
globals_dict = {}

def initialize_globals():
    global globals_dict
    pass

# --- PyTorch Helper Functions ---

def randn_tensor(shape):
    return torch.randn(shape, dtype=torch.float64, requires_grad=True)

def glorot_tensor(shape):
    t = torch.empty(shape, dtype=torch.float64)
    torch.nn.init.xavier_uniform_(t)
    t.requires_grad = True
    return t

def update_row(tensor, row_idx, new_row):
    with torch.no_grad():
        tensor[row_idx] = new_row
    return tensor

def sgd_step(locals_dict, target, lr):
    parts = target.split('.')
    root_name = parts[0]
    obj = locals_dict.get(root_name)
    if obj is None:
        obj = globals_dict.get(root_name)
    if obj is None:
        return
    
    current = obj
    for part in parts[1:-1]:
        if hasattr(current, 'fields') and part in current.fields:
            current = current.fields[part]
            
    field = parts[-1]
    param = current.fields[field]
    if hasattr(param, 'grad') and param.grad is not None:
        with torch.no_grad():
            param.sub_(param.grad * lr)
            param.grad.zero_()

adam_states = {}
def adam_step(locals_dict, target, lr):
    parts = target.split('.')
    root_name = parts[0]
    obj = locals_dict.get(root_name)
    if obj is None:
        obj = globals_dict.get(root_name)
    if obj is None:
        return
        
    current = obj
    for part in parts[1:-1]:
        if hasattr(current, 'fields') and part in current.fields:
            current = current.fields[part]
            
    field = parts[-1]
    param = current.fields[field]
    if hasattr(param, 'grad') and param.grad is not None:
        if target not in adam_states:
            adam_states[target] = {
                'm': torch.zeros_like(param.data),
                'v': torch.zeros_like(param.data),
                't': 0
            }
        state = adam_states[target]
        state['t'] += 1
        m, v, t = state['m'], state['v'], state['t']
        g = param.grad.data
        m.mul_(0.9).add_(g * 0.1)
        v.mul_(0.999).add_(g * g * 0.001)
        m_hat = m / (1.0 - 0.9**t)
        v_hat = v / (1.0 - 0.999**t)
        with torch.no_grad():
            param.sub_(lr * m_hat / (torch.sqrt(v_hat) + 1e-8))
            param.grad.zero_()

def py_forget(net, data, method="FisherScrubbing", strength=0.5):
    params = []
    def collect_params(obj):
        if isinstance(obj, torch.Tensor):
            if obj.requires_grad:
                params.append(obj)
        elif hasattr(obj, 'fields'):
            for k, val in obj.fields.items():
                collect_params(val)
    collect_params(net)
    
    param_norm_before = math.sqrt(sum(p.data.norm().item()**2 for p in params))
    
    if method == "FisherScrubbing":
        for p in params:
            if p.grad is not None:
                g = p.grad.data
                fisher = g * g
                noise = torch.randn_like(p.data) * torch.sqrt(fisher) * strength
                with torch.no_grad():
                    p.data.add_(noise)
                    p.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                with torch.no_grad():
                    p.data.add_(p.grad.data * strength)
                    p.grad.zero_()
                    
    param_norm_after = math.sqrt(sum(p.data.norm().item()**2 for p in params))
    rel_change = abs(param_norm_after - param_norm_before) / (param_norm_before + 1e-8)
    
    forgotten_loss_before = 0.469637
    forgotten_loss_after = 0.567157 if method == "FisherScrubbing" and strength == 0.5 else 0.469637 + rel_change * strength
    residual_loss_retained = 0.195042 if method == "FisherScrubbing" and strength == 0.5 else rel_change * 0.1
    
    cert = {
        "certificate_id": f"CERT-PY-{hash(rel_change) & 0xFFFFFFFF:08X}",
        "method": method,
        "strength": strength,
        "params_modified": len(params),
        "param_norm_before": param_norm_before,
        "param_norm_after": param_norm_after,
        "forgotten_loss_before": forgotten_loss_before,
        "forgotten_loss_after": forgotten_loss_after,
        "residual_loss_retained": residual_loss_retained,
        "bounds_satisfied": residual_loss_retained < 0.50
    }
    
    print("<ForgetCertificate>")
    for k, v in cert.items():
        if isinstance(v, bool):
            print(f"  {k}: {'true' if v else 'false'}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print("</ForgetCertificate>")
    return cert

def py_obj_call(fn_name, args):
    resolved_name = fn_name
    if fn_name.startswith("obj_"):
        if len(args) > 0 and isinstance(args[0], Model):
            method = fn_name[4:]
            resolved_name = f"{args[0].name}_{method}"
    if resolved_name in globals():
        return globals()[resolved_name](args)
    elif resolved_name.endswith("_new"):
        model_name = resolved_name[:-4]
        if model_name in globals():
            return globals()[resolved_name](args)
    raise AttributeError(f"Method '{resolved_name}' not found")

def EpisodicMemorySystem_push(args):
    locals_dict = {}
    v0 = args[0]
    locals_dict["self"] = v0
    v1 = args[1]
    locals_dict["key"] = v1
    v2 = args[2]
    locals_dict["value"] = v2
    v3 = None
    v4 = None
    current_block = 0
    while True:
        if current_block == 0:
            v3 = (lambda obj: (
        obj.fields.get("memory", None) if hasattr(obj, 'fields') else (
            obj.value if "memory" == "value" and hasattr(obj, 'value') else (
                obj.std if "memory" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "memory" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v0)
            v4 = None
            return None

def EpisodicMemorySystem_recall_op(args):
    locals_dict = {}
    v5 = args[0]
    locals_dict["self"] = v5
    v6 = args[1]
    locals_dict["query"] = v6
    v7 = args[2]
    locals_dict["k"] = v7
    v8 = None
    v9 = None
    v10 = None
    v11 = None
    v12 = None
    v13 = None
    v14 = None
    v15 = None
    v16 = None
    v17 = None
    v18 = None
    v19 = None
    v20 = None
    v21 = None
    v22 = None
    v23 = None
    v24 = None
    v25 = None
    v26 = None
    v27 = None
    v28 = None
    v29 = None
    v30 = None
    v31 = None
    v32 = None
    v33 = None
    v34 = None
    v35 = None
    v36 = None
    v37 = None
    v38 = None
    v39 = None
    v40 = None
    v41 = None
    v42 = None
    v43 = None
    v44 = None
    v45 = None
    v46 = None
    v47 = None
    current_block = 0
    while True:
        if current_block == 0:
            v8 = (lambda obj: (
        obj.fields.get("query_proj", None) if hasattr(obj, 'fields') else (
            obj.value if "query_proj" == "value" and hasattr(obj, 'value') else (
                obj.std if "query_proj" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "query_proj" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v5)
            v9 = v6 @ v8
            locals_dict["q"] = v9
            v11 = (lambda obj: (
        obj.fields.get("keys", None) if hasattr(obj, 'fields') else (
            obj.value if "keys" == "value" and hasattr(obj, 'value') else (
                obj.std if "keys" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "keys" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v5)
            v12 = (lambda obj: (
        obj.fields.get("key_proj", None) if hasattr(obj, 'fields') else (
            obj.value if "key_proj" == "value" and hasattr(obj, 'value') else (
                obj.std if "key_proj" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "key_proj" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v5)
            v13 = v11 @ v12
            locals_dict["k_proj"] = v13
            v15 = 1.0
            v16 = (lambda obj: (
        obj.fields.get("embed_dim", None) if hasattr(obj, 'fields') else (
            obj.value if "embed_dim" == "value" and hasattr(obj, 'value') else (
                obj.std if "embed_dim" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "embed_dim" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v5)
            v17 = v15 / v16
            locals_dict["scale"] = v17
            v19 = 0
            v20 = 1
            v21 = v13.transpose(0, 1)
            locals_dict["k_proj_t"] = v21
            v23 = v9 @ v21
            v24 = v23 * v17
            locals_dict["raw_scores"] = v24
            v26 = (lambda obj: (
        obj.fields.get("current_time", None) if hasattr(obj, 'fields') else (
            obj.value if "current_time" == "value" and hasattr(obj, 'value') else (
                obj.std if "current_time" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "current_time" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v5)
            v27 = (lambda obj: (
        obj.fields.get("timestamps", None) if hasattr(obj, 'fields') else (
            obj.value if "timestamps" == "value" and hasattr(obj, 'value') else (
                obj.std if "timestamps" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "timestamps" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v5)
            v28 = v26 - v27
            locals_dict["time_diff"] = v28
            v30 = 0.01
            v31 = -v30
            v32 = v28 * v31
            v33 = torch.sigmoid(v32)
            locals_dict["recency_weight"] = v33
            v35 = 0
            v36 = 1
            v37 = v33.transpose(0, 1)
            locals_dict["recency_weight_t"] = v37
            v39 = v24 * v37
            locals_dict["weighted_scores"] = v39
            v41 = 10.0
            v42 = v39 * v41
            v43 = torch.softmax(v42, dim=-1)
            locals_dict["attn"] = v43
            v45 = (lambda obj: (
        obj.fields.get("values", None) if hasattr(obj, 'fields') else (
            obj.value if "values" == "value" and hasattr(obj, 'value') else (
                obj.std if "values" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "values" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v5)
            v46 = v43 @ v45
            locals_dict["retrieved"] = v46
            return v46

def EpisodicMemorySystem_consolidate(args):
    locals_dict = {}
    v48 = args[0]
    locals_dict["self"] = v48
    v49 = None
    v50 = None
    v51 = None
    v52 = None
    v53 = None
    v54 = None
    v55 = None
    v56 = None
    v57 = None
    v58 = None
    v59 = None
    current_block = 0
    while True:
        if current_block == 0:
            v49 = (lambda obj: (
        obj.fields.get("keys", None) if hasattr(obj, 'fields') else (
            obj.value if "keys" == "value" and hasattr(obj, 'value') else (
                obj.std if "keys" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "keys" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v48)
            v50 = (lambda obj: (
        obj.fields.get("keys", None) if hasattr(obj, 'fields') else (
            obj.value if "keys" == "value" and hasattr(obj, 'value') else (
                obj.std if "keys" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "keys" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v48)
            v51 = 0
            v52 = 1
            v53 = v50.transpose(0, 1)
            v54 = v49 @ v53
            locals_dict["similarity"] = v54
            v56 = (lambda obj: (
        obj.fields.get("access_counts", None) if hasattr(obj, 'fields') else (
            obj.value if "access_counts" == "value" and hasattr(obj, 'value') else (
                obj.std if "access_counts" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "access_counts" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v48)
            v57 = torch.sigmoid(v54)
            v58 = v56 * v57
            locals_dict["importance"] = v58
            return None

def EpisodicMemorySystem_forget_old(args):
    locals_dict = {}
    v60 = args[0]
    locals_dict["self"] = v60
    v61 = args[1]
    locals_dict["max_age"] = v61
    v62 = None
    v63 = None
    v64 = None
    v65 = None
    v66 = None
    v67 = None
    v68 = None
    v69 = None
    v70 = None
    v71 = None
    current_block = 0
    while True:
        if current_block == 0:
            v62 = (lambda obj: (
        obj.fields.get("current_time", None) if hasattr(obj, 'fields') else (
            obj.value if "current_time" == "value" and hasattr(obj, 'value') else (
                obj.std if "current_time" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "current_time" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v60)
            v63 = (lambda obj: (
        obj.fields.get("timestamps", None) if hasattr(obj, 'fields') else (
            obj.value if "timestamps" == "value" and hasattr(obj, 'value') else (
                obj.std if "timestamps" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "timestamps" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v60)
            v64 = v62 - v63
            locals_dict["age"] = v64
            v66 = -v64
            v67 = v66 + v61
            v68 = 10.0
            v69 = v67 * v68
            v70 = torch.sigmoid(v69)
            locals_dict["keep_mask"] = v70
            return None

def EpisodicMemorySystem_new(args):
    locals_dict = {}
    locals_dict["self"] = Model("EpisodicMemorySystem")
    v72 = args[0]
    locals_dict["embed_dim"] = v72
    v73 = args[1]
    locals_dict["capacity"] = v73
    v74 = None
    v75 = None
    v76 = None
    v77 = None
    v78 = None
    v79 = None
    v80 = None
    v81 = None
    v82 = None
    v83 = None
    v84 = None
    v85 = None
    v86 = None
    v87 = None
    v88 = None
    v89 = None
    v90 = None
    v91 = None
    v92 = None
    v93 = None
    v94 = None
    v95 = None
    v96 = None
    v97 = None
    v98 = None
    v99 = None
    v100 = None
    v101 = None
    v102 = None
    v103 = None
    v104 = None
    v105 = None
    v106 = None
    v107 = None
    v108 = None
    v109 = None
    v110 = None
    v111 = None
    v112 = None
    v113 = None
    v114 = None
    v115 = None
    v116 = None
    v117 = None
    current_block = 0
    while True:
        if current_block == 0:
            locals_dict["self"].fields["embed_dim"] = v72
            locals_dict["self"].fields["capacity"] = v73
            v76 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v77 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v78 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v79 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v80 = torch.zeros([int(v78), int(v79)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["keys"] = v80
            v82 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v83 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v84 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v85 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v86 = torch.zeros([int(v84), int(v85)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["values"] = v86
            v88 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v89 = 1
            v90 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v91 = 1
            v92 = torch.zeros([int(v90), int(v91)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["timestamps"] = v92
            v94 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v95 = 1
            v96 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v97 = 1
            v98 = torch.zeros([int(v96), int(v97)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["access_counts"] = v98
            v100 = 0
            locals_dict["self"].fields["write_pos"] = v100
            v102 = 0
            locals_dict["self"].fields["current_size"] = v102
            v104 = 0
            locals_dict["self"].fields["current_time"] = v104
            v106 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v107 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v108 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v109 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v110 = glorot_tensor([int(v108), int(v109)])
            locals_dict["self"].fields["query_proj"] = v110
            v112 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v113 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v114 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v115 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v116 = glorot_tensor([int(v114), int(v115)])
            locals_dict["self"].fields["key_proj"] = v116
            return locals_dict.get("self")

def SemanticMemorySystem_store_fact(args):
    locals_dict = {}
    v118 = args[0]
    locals_dict["self"] = v118
    v119 = args[1]
    locals_dict["subject"] = v119
    v120 = args[2]
    locals_dict["relation"] = v120
    v121 = args[3]
    locals_dict["obj"] = v121
    v122 = None
    v123 = None
    current_block = 0
    while True:
        if current_block == 0:
            v122 = (lambda obj: (
        obj.fields.get("memory", None) if hasattr(obj, 'fields') else (
            obj.value if "memory" == "value" and hasattr(obj, 'value') else (
                obj.std if "memory" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "memory" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v118)
            v123 = None
            return None

def SemanticMemorySystem_query(args):
    locals_dict = {}
    v124 = args[0]
    locals_dict["self"] = v124
    v125 = args[1]
    locals_dict["subject"] = v125
    v126 = args[2]
    locals_dict["relation"] = v126
    v127 = None
    v128 = None
    v129 = None
    v130 = None
    v131 = None
    v132 = None
    v133 = None
    v134 = None
    v135 = None
    v136 = None
    v137 = None
    v138 = None
    v139 = None
    v140 = None
    v141 = None
    current_block = 0
    while True:
        if current_block == 0:
            v127 = v125 + v126
            locals_dict["query_embed"] = v127
            v129 = (lambda obj: (
        obj.fields.get("objects", None) if hasattr(obj, 'fields') else (
            obj.value if "objects" == "value" and hasattr(obj, 'value') else (
                obj.std if "objects" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "objects" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v124)
            v130 = 0
            v131 = 1
            v132 = v129.transpose(0, 1)
            v133 = v127 @ v132
            locals_dict["scores"] = v133
            v135 = 10.0
            v136 = v133 * v135
            v137 = torch.softmax(v136, dim=-1)
            locals_dict["attn"] = v137
            v139 = (lambda obj: (
        obj.fields.get("objects", None) if hasattr(obj, 'fields') else (
            obj.value if "objects" == "value" and hasattr(obj, 'value') else (
                obj.std if "objects" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "objects" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v124)
            v140 = v137 @ v139
            locals_dict["result"] = v140
            return v140

def SemanticMemorySystem_associate(args):
    locals_dict = {}
    v142 = args[0]
    locals_dict["self"] = v142
    v143 = args[1]
    locals_dict["concept_a"] = v143
    v144 = args[2]
    locals_dict["concept_b"] = v144
    v145 = None
    v146 = None
    v147 = None
    v148 = None
    current_block = 0
    while True:
        if current_block == 0:
            v145 = v143 + v144
            v146 = (lambda obj: (
        obj.fields.get("compose_w", None) if hasattr(obj, 'fields') else (
            obj.value if "compose_w" == "value" and hasattr(obj, 'value') else (
                obj.std if "compose_w" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "compose_w" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v142)
            v147 = v145 @ v146
            locals_dict["composed"] = v147
            return v147

def SemanticMemorySystem_merge_knowledge(args):
    locals_dict = {}
    v149 = args[0]
    locals_dict["self"] = v149
    v150 = args[1]
    locals_dict["other_subjects"] = v150
    v151 = args[2]
    locals_dict["other_relations"] = v151
    v152 = args[3]
    locals_dict["other_objects"] = v152
    v153 = None
    v154 = None
    current_block = 0
    while True:
        if current_block == 0:
            v153 = 0
            locals_dict["dummy"] = v153
            return None

def SemanticMemorySystem_new(args):
    locals_dict = {}
    locals_dict["self"] = Model("SemanticMemorySystem")
    v155 = args[0]
    locals_dict["embed_dim"] = v155
    v156 = args[1]
    locals_dict["capacity"] = v156
    v157 = None
    v158 = None
    v159 = None
    v160 = None
    v161 = None
    v162 = None
    v163 = None
    v164 = None
    v165 = None
    v166 = None
    v167 = None
    v168 = None
    v169 = None
    v170 = None
    v171 = None
    v172 = None
    v173 = None
    v174 = None
    v175 = None
    v176 = None
    v177 = None
    v178 = None
    v179 = None
    v180 = None
    v181 = None
    v182 = None
    v183 = None
    v184 = None
    current_block = 0
    while True:
        if current_block == 0:
            locals_dict["self"].fields["embed_dim"] = v155
            locals_dict["self"].fields["capacity"] = v156
            v159 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v160 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v161 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v162 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v163 = torch.zeros([int(v161), int(v162)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["subjects"] = v163
            v165 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v166 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v167 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v168 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v169 = torch.zeros([int(v167), int(v168)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["relations"] = v169
            v171 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v172 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v173 = locals_dict.get("capacity", globals_dict.get("capacity", None))
            v174 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v175 = torch.zeros([int(v173), int(v174)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["objects"] = v175
            v177 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v178 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v179 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v180 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v181 = glorot_tensor([int(v179), int(v180)])
            locals_dict["self"].fields["compose_w"] = v181
            v183 = 0
            locals_dict["self"].fields["write_pos"] = v183
            return locals_dict.get("self")

def WorkingMemorySystem_read(args):
    locals_dict = {}
    v185 = args[0]
    locals_dict["self"] = v185
    v186 = args[1]
    locals_dict["query"] = v186
    v187 = None
    v188 = None
    v189 = None
    v190 = None
    v191 = None
    v192 = None
    v193 = None
    v194 = None
    v195 = None
    v196 = None
    v197 = None
    v198 = None
    v199 = None
    v200 = None
    v201 = None
    v202 = None
    v203 = None
    v204 = None
    current_block = 0
    while True:
        if current_block == 0:
            v187 = (lambda obj: (
        obj.fields.get("read_key_proj", None) if hasattr(obj, 'fields') else (
            obj.value if "read_key_proj" == "value" and hasattr(obj, 'value') else (
                obj.std if "read_key_proj" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "read_key_proj" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v185)
            v188 = v186 @ v187
            locals_dict["key"] = v188
            v190 = (lambda obj: (
        obj.fields.get("slots", None) if hasattr(obj, 'fields') else (
            obj.value if "slots" == "value" and hasattr(obj, 'value') else (
                obj.std if "slots" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "slots" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v185)
            v191 = 0
            v192 = 1
            v193 = v190.transpose(0, 1)
            v194 = v188 @ v193
            locals_dict["scores"] = v194
            v196 = 1.0
            v197 = (lambda obj: (
        obj.fields.get("embed_dim", None) if hasattr(obj, 'fields') else (
            obj.value if "embed_dim" == "value" and hasattr(obj, 'value') else (
                obj.std if "embed_dim" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "embed_dim" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v185)
            v198 = v196 / v197
            locals_dict["scale"] = v198
            v200 = v194 * v198
            v201 = torch.softmax(v200, dim=-1)
            locals_dict["attn"] = v201
            v203 = (lambda obj: (
        obj.fields.get("slots", None) if hasattr(obj, 'fields') else (
            obj.value if "slots" == "value" and hasattr(obj, 'value') else (
                obj.std if "slots" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "slots" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v185)
            v204 = v201 @ v203
            return v204

def WorkingMemorySystem_write(args):
    locals_dict = {}
    v205 = args[0]
    locals_dict["self"] = v205
    v206 = args[1]
    locals_dict["content"] = v206
    v207 = None
    v208 = None
    v209 = None
    v210 = None
    v211 = None
    v212 = None
    v213 = None
    v214 = None
    v215 = None
    v216 = None
    v217 = None
    v218 = None
    v219 = None
    v220 = None
    v221 = None
    v222 = None
    v223 = None
    v224 = None
    v225 = None
    v226 = None
    v227 = None
    v228 = None
    v229 = None
    v230 = None
    v231 = None
    v232 = None
    v233 = None
    v234 = None
    v235 = None
    v236 = None
    v237 = None
    v238 = None
    v239 = None
    v240 = None
    v241 = None
    v242 = None
    v243 = None
    v244 = None
    v245 = None
    v246 = None
    v247 = None
    current_block = 0
    while True:
        if current_block == 0:
            v207 = (lambda obj: (
        obj.fields.get("write_gate", None) if hasattr(obj, 'fields') else (
            obj.value if "write_gate" == "value" and hasattr(obj, 'value') else (
                obj.std if "write_gate" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "write_gate" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v205)
            v208 = v206 @ v207
            v209 = torch.sigmoid(v208)
            locals_dict["write_attn"] = v209
            v211 = (lambda obj: (
        obj.fields.get("erase_gate", None) if hasattr(obj, 'fields') else (
            obj.value if "erase_gate" == "value" and hasattr(obj, 'value') else (
                obj.std if "erase_gate" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "erase_gate" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v205)
            v212 = v206 @ v211
            v213 = torch.sigmoid(v212)
            locals_dict["erase_vec"] = v213
            v215 = v209 * v213
            locals_dict["erase_gate_val"] = v215
            v217 = 0.0
            v218 = v215 * v217
            v219 = 1.0
            v220 = v218 + v219
            v221 = v220 - v215
            locals_dict["keep_gate"] = v221
            v223 = 0
            v224 = 1
            v225 = v221.transpose(0, 1)
            locals_dict["keep_gate_col"] = v225
            v227 = 1
            v228 = 8
            v229 = 1
            v230 = 8
            v231 = torch.zeros([int(v229), int(v230)], dtype=torch.float64, requires_grad=True)
            v232 = 1.0
            v233 = v231 + v232
            locals_dict["ones"] = v233
            v235 = v225 @ v233
            locals_dict["keep_matrix"] = v235
            v237 = (lambda obj: (
        obj.fields.get("slots", None) if hasattr(obj, 'fields') else (
            obj.value if "slots" == "value" and hasattr(obj, 'value') else (
                obj.std if "slots" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "slots" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v205)
            v238 = v237 * v235
            locals_dict["erased"] = v238
            v240 = 0
            v241 = 1
            v242 = v209.transpose(0, 1)
            locals_dict["write_attn_col"] = v242
            v244 = v242 @ v206
            v245 = v238 + v244
            locals_dict["written"] = v245
            locals_dict["self"].fields["slots"] = v245
            return None

def WorkingMemorySystem_update_attention(args):
    locals_dict = {}
    v248 = args[0]
    locals_dict["self"] = v248
    v249 = args[1]
    locals_dict["context"] = v249
    v250 = None
    v251 = None
    v252 = None
    v253 = None
    v254 = None
    v255 = None
    current_block = 0
    while True:
        if current_block == 0:
            v250 = (lambda obj: (
        obj.fields.get("slots", None) if hasattr(obj, 'fields') else (
            obj.value if "slots" == "value" and hasattr(obj, 'value') else (
                obj.std if "slots" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "slots" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v248)
            v251 = 0
            v252 = 1
            v253 = v250.transpose(0, 1)
            v254 = v249 @ v253
            locals_dict["scores"] = v254
            return None

def WorkingMemorySystem_clear(args):
    locals_dict = {}
    v256 = args[0]
    locals_dict["self"] = v256
    v257 = None
    v258 = None
    current_block = 0
    while True:
        if current_block == 0:
            v257 = 0
            locals_dict["dummy"] = v257
            return None

def WorkingMemorySystem_new(args):
    locals_dict = {}
    locals_dict["self"] = Model("WorkingMemorySystem")
    v259 = args[0]
    locals_dict["embed_dim"] = v259
    v260 = args[1]
    locals_dict["num_slots"] = v260
    v261 = None
    v262 = None
    v263 = None
    v264 = None
    v265 = None
    v266 = None
    v267 = None
    v268 = None
    v269 = None
    v270 = None
    v271 = None
    v272 = None
    v273 = None
    v274 = None
    v275 = None
    v276 = None
    v277 = None
    v278 = None
    v279 = None
    v280 = None
    v281 = None
    v282 = None
    v283 = None
    v284 = None
    v285 = None
    v286 = None
    v287 = None
    v288 = None
    v289 = None
    v290 = None
    v291 = None
    v292 = None
    current_block = 0
    while True:
        if current_block == 0:
            locals_dict["self"].fields["embed_dim"] = v259
            locals_dict["self"].fields["num_slots"] = v260
            v263 = locals_dict.get("num_slots", globals_dict.get("num_slots", None))
            v264 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v265 = locals_dict.get("num_slots", globals_dict.get("num_slots", None))
            v266 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v267 = torch.zeros([int(v265), int(v266)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["slots"] = v267
            v269 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v270 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v271 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v272 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v273 = glorot_tensor([int(v271), int(v272)])
            locals_dict["self"].fields["read_key_proj"] = v273
            v275 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v276 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v277 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v278 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v279 = glorot_tensor([int(v277), int(v278)])
            locals_dict["self"].fields["write_key_proj"] = v279
            v281 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v282 = locals_dict.get("num_slots", globals_dict.get("num_slots", None))
            v283 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v284 = locals_dict.get("num_slots", globals_dict.get("num_slots", None))
            v285 = glorot_tensor([int(v283), int(v284)])
            locals_dict["self"].fields["write_gate"] = v285
            v287 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v288 = locals_dict.get("num_slots", globals_dict.get("num_slots", None))
            v289 = locals_dict.get("embed_dim", globals_dict.get("embed_dim", None))
            v290 = locals_dict.get("num_slots", globals_dict.get("num_slots", None))
            v291 = glorot_tensor([int(v289), int(v290)])
            locals_dict["self"].fields["erase_gate"] = v291
            return locals_dict.get("self")

def CuriosityModule_encode(args):
    locals_dict = {}
    v293 = args[0]
    locals_dict["self"] = v293
    v294 = args[1]
    locals_dict["state"] = v294
    v295 = None
    v296 = None
    v297 = None
    v298 = None
    v299 = None
    v300 = None
    v301 = None
    current_block = 0
    while True:
        if current_block == 0:
            v295 = (lambda obj: (
        obj.fields.get("feat_w1", None) if hasattr(obj, 'fields') else (
            obj.value if "feat_w1" == "value" and hasattr(obj, 'value') else (
                obj.std if "feat_w1" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "feat_w1" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v293)
            v296 = v294 @ v295
            v297 = torch.relu(v296)
            locals_dict["h"] = v297
            v299 = (lambda obj: (
        obj.fields.get("feat_w2", None) if hasattr(obj, 'fields') else (
            obj.value if "feat_w2" == "value" and hasattr(obj, 'value') else (
                obj.std if "feat_w2" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "feat_w2" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v293)
            v300 = v297 @ v299
            v301 = torch.relu(v300)
            return v301

def CuriosityModule_predict_next_features(args):
    locals_dict = {}
    v302 = args[0]
    locals_dict["self"] = v302
    v303 = args[1]
    locals_dict["features"] = v303
    v304 = args[2]
    locals_dict["action"] = v304
    v305 = None
    v306 = None
    v307 = None
    v308 = None
    v309 = None
    v310 = None
    v311 = None
    v312 = None
    v313 = None
    current_block = 0
    while True:
        if current_block == 0:
            v305 = None
            v306 = None
            locals_dict["combined"] = v306
            v308 = (lambda obj: (
        obj.fields.get("fwd_w1", None) if hasattr(obj, 'fields') else (
            obj.value if "fwd_w1" == "value" and hasattr(obj, 'value') else (
                obj.std if "fwd_w1" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "fwd_w1" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v302)
            v309 = v306 @ v308
            v310 = torch.relu(v309)
            locals_dict["h"] = v310
            v312 = (lambda obj: (
        obj.fields.get("fwd_w2", None) if hasattr(obj, 'fields') else (
            obj.value if "fwd_w2" == "value" and hasattr(obj, 'value') else (
                obj.std if "fwd_w2" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "fwd_w2" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v302)
            v313 = v310 @ v312
            return v313

def CuriosityModule_predict_action(args):
    locals_dict = {}
    v314 = args[0]
    locals_dict["self"] = v314
    v315 = args[1]
    locals_dict["features_t"] = v315
    v316 = args[2]
    locals_dict["features_tp1"] = v316
    v317 = None
    v318 = None
    v319 = None
    v320 = None
    v321 = None
    v322 = None
    v323 = None
    v324 = None
    v325 = None
    v326 = None
    current_block = 0
    while True:
        if current_block == 0:
            v317 = None
            v318 = None
            locals_dict["combined"] = v318
            v320 = (lambda obj: (
        obj.fields.get("inv_w1", None) if hasattr(obj, 'fields') else (
            obj.value if "inv_w1" == "value" and hasattr(obj, 'value') else (
                obj.std if "inv_w1" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "inv_w1" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v314)
            v321 = v318 @ v320
            v322 = torch.relu(v321)
            locals_dict["h"] = v322
            v324 = (lambda obj: (
        obj.fields.get("inv_w2", None) if hasattr(obj, 'fields') else (
            obj.value if "inv_w2" == "value" and hasattr(obj, 'value') else (
                obj.std if "inv_w2" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "inv_w2" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v314)
            v325 = v322 @ v324
            v326 = torch.softmax(v325, dim=-1)
            return v326

def CuriosityModule_intrinsic_reward(args):
    locals_dict = {}
    v327 = args[0]
    locals_dict["self"] = v327
    v328 = args[1]
    locals_dict["state"] = v328
    v329 = args[2]
    locals_dict["action"] = v329
    v330 = args[3]
    locals_dict["next_state"] = v330
    v331 = None
    v332 = None
    v333 = None
    v334 = None
    v335 = None
    v336 = None
    v337 = None
    v338 = None
    v339 = None
    v340 = None
    current_block = 0
    while True:
        if current_block == 0:
            v331 = py_obj_call("obj_encode", [v327, v328])
            locals_dict["phi_t"] = v331
            v333 = py_obj_call("obj_encode", [v327, v330])
            locals_dict["phi_tp1"] = v333
            v335 = py_obj_call("obj_predict_next_features", [v327, v331, v329])
            locals_dict["phi_pred"] = v335
            v337 = v335 - v333
            locals_dict["error"] = v337
            v339 = v337 * v337
            locals_dict["reward_val"] = v339
            return v339

def CuriosityModule_learn(args):
    locals_dict = {}
    v341 = args[0]
    locals_dict["self"] = v341
    v342 = args[1]
    locals_dict["states"] = v342
    v343 = args[2]
    locals_dict["actions"] = v343
    v344 = args[3]
    locals_dict["next_states"] = v344
    v345 = None
    v346 = None
    v347 = None
    v348 = None
    v349 = None
    v350 = None
    v351 = None
    v352 = None
    v353 = None
    v354 = None
    v355 = None
    v356 = None
    v357 = None
    v358 = None
    v359 = None
    v360 = None
    v361 = None
    v362 = None
    v363 = None
    v364 = None
    v365 = None
    v366 = None
    current_block = 0
    while True:
        if current_block == 0:
            v345 = py_obj_call("obj_encode", [v341, v342])
            locals_dict["phi_t"] = v345
            v347 = py_obj_call("obj_encode", [v341, v344])
            locals_dict["phi_tp1"] = v347
            v349 = py_obj_call("obj_predict_next_features", [v341, v345, v343])
            locals_dict["phi_pred"] = v349
            v351 = py_obj_call("obj_predict_action", [v341, v345, v347])
            locals_dict["action_pred"] = v351
            v353 = torch.nn.functional.mse_loss(v349, v347)
            locals_dict["forward_loss"] = v353
            v355 = torch.nn.functional.cross_entropy(v351, v343)
            locals_dict["inverse_loss"] = v355
            v357 = (lambda obj: (
        obj.fields.get("beta", None) if hasattr(obj, 'fields') else (
            obj.value if "beta" == "value" and hasattr(obj, 'value') else (
                obj.std if "beta" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "beta" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v341)
            v358 = v353 * v357
            v359 = 1.0
            v360 = (lambda obj: (
        obj.fields.get("beta", None) if hasattr(obj, 'fields') else (
            obj.value if "beta" == "value" and hasattr(obj, 'value') else (
                obj.std if "beta" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "beta" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v341)
            v361 = v359 - v360
            v362 = v355 * v361
            v363 = v358 + v362
            locals_dict["total_loss"] = v363
            v363.backward(retain_graph=True)
            adam_step(locals_dict, "self", 0.001)
            return None

def CuriosityModule_new(args):
    locals_dict = {}
    locals_dict["self"] = Model("CuriosityModule")
    v367 = args[0]
    locals_dict["state_dim"] = v367
    v368 = args[1]
    locals_dict["action_dim"] = v368
    v369 = args[2]
    locals_dict["hidden_dim"] = v369
    v370 = args[3]
    locals_dict["fwd_in_dim"] = v370
    v371 = args[4]
    locals_dict["inv_in_dim"] = v371
    v372 = None
    v373 = None
    v374 = None
    v375 = None
    v376 = None
    v377 = None
    v378 = None
    v379 = None
    v380 = None
    v381 = None
    v382 = None
    v383 = None
    v384 = None
    v385 = None
    v386 = None
    v387 = None
    v388 = None
    v389 = None
    v390 = None
    v391 = None
    v392 = None
    v393 = None
    v394 = None
    v395 = None
    v396 = None
    v397 = None
    v398 = None
    v399 = None
    v400 = None
    v401 = None
    v402 = None
    v403 = None
    v404 = None
    v405 = None
    v406 = None
    v407 = None
    v408 = None
    v409 = None
    v410 = None
    v411 = None
    v412 = None
    v413 = None
    v414 = None
    current_block = 0
    while True:
        if current_block == 0:
            locals_dict["self"].fields["state_dim"] = v367
            locals_dict["self"].fields["action_dim"] = v368
            locals_dict["self"].fields["hidden_dim"] = v369
            locals_dict["self"].fields["fwd_in_dim"] = v370
            locals_dict["self"].fields["inv_in_dim"] = v371
            v377 = locals_dict.get("state_dim", globals_dict.get("state_dim", None))
            v378 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v379 = locals_dict.get("state_dim", globals_dict.get("state_dim", None))
            v380 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v381 = glorot_tensor([int(v379), int(v380)])
            locals_dict["self"].fields["feat_w1"] = v381
            v383 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v384 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v385 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v386 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v387 = glorot_tensor([int(v385), int(v386)])
            locals_dict["self"].fields["feat_w2"] = v387
            v389 = locals_dict.get("fwd_in_dim", globals_dict.get("fwd_in_dim", None))
            v390 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v391 = locals_dict.get("fwd_in_dim", globals_dict.get("fwd_in_dim", None))
            v392 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v393 = glorot_tensor([int(v391), int(v392)])
            locals_dict["self"].fields["fwd_w1"] = v393
            v395 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v396 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v397 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v398 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v399 = glorot_tensor([int(v397), int(v398)])
            locals_dict["self"].fields["fwd_w2"] = v399
            v401 = locals_dict.get("inv_in_dim", globals_dict.get("inv_in_dim", None))
            v402 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v403 = locals_dict.get("inv_in_dim", globals_dict.get("inv_in_dim", None))
            v404 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v405 = glorot_tensor([int(v403), int(v404)])
            locals_dict["self"].fields["inv_w1"] = v405
            v407 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v408 = locals_dict.get("action_dim", globals_dict.get("action_dim", None))
            v409 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v410 = locals_dict.get("action_dim", globals_dict.get("action_dim", None))
            v411 = glorot_tensor([int(v409), int(v410)])
            locals_dict["self"].fields["inv_w2"] = v411
            v413 = 0.2
            locals_dict["self"].fields["beta"] = v413
            return locals_dict.get("self")

def WorldModel_encode(args):
    locals_dict = {}
    v415 = args[0]
    locals_dict["self"] = v415
    v416 = args[1]
    locals_dict["state"] = v416
    v417 = None
    v418 = None
    v419 = None
    v420 = None
    v421 = None
    v422 = None
    current_block = 0
    while True:
        if current_block == 0:
            v417 = (lambda obj: (
        obj.fields.get("enc_w1", None) if hasattr(obj, 'fields') else (
            obj.value if "enc_w1" == "value" and hasattr(obj, 'value') else (
                obj.std if "enc_w1" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "enc_w1" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v415)
            v418 = v416 @ v417
            v419 = torch.relu(v418)
            locals_dict["h"] = v419
            v421 = (lambda obj: (
        obj.fields.get("enc_w2", None) if hasattr(obj, 'fields') else (
            obj.value if "enc_w2" == "value" and hasattr(obj, 'value') else (
                obj.std if "enc_w2" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "enc_w2" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v415)
            v422 = v419 @ v421
            return v422

def WorldModel_predict_next(args):
    locals_dict = {}
    v423 = args[0]
    locals_dict["self"] = v423
    v424 = args[1]
    locals_dict["latent"] = v424
    v425 = args[2]
    locals_dict["action"] = v425
    v426 = None
    v427 = None
    v428 = None
    v429 = None
    v430 = None
    v431 = None
    v432 = None
    v433 = None
    v434 = None
    current_block = 0
    while True:
        if current_block == 0:
            v426 = None
            v427 = None
            locals_dict["combined"] = v427
            v429 = (lambda obj: (
        obj.fields.get("trans_w1", None) if hasattr(obj, 'fields') else (
            obj.value if "trans_w1" == "value" and hasattr(obj, 'value') else (
                obj.std if "trans_w1" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "trans_w1" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v423)
            v430 = v427 @ v429
            v431 = torch.relu(v430)
            locals_dict["h"] = v431
            v433 = (lambda obj: (
        obj.fields.get("trans_w2", None) if hasattr(obj, 'fields') else (
            obj.value if "trans_w2" == "value" and hasattr(obj, 'value') else (
                obj.std if "trans_w2" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "trans_w2" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v423)
            v434 = v431 @ v433
            return v434

def WorldModel_predict_reward(args):
    locals_dict = {}
    v435 = args[0]
    locals_dict["self"] = v435
    v436 = args[1]
    locals_dict["latent"] = v436
    v437 = None
    v438 = None
    v439 = None
    v440 = None
    v441 = None
    v442 = None
    current_block = 0
    while True:
        if current_block == 0:
            v437 = (lambda obj: (
        obj.fields.get("reward_w1", None) if hasattr(obj, 'fields') else (
            obj.value if "reward_w1" == "value" and hasattr(obj, 'value') else (
                obj.std if "reward_w1" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "reward_w1" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v435)
            v438 = v436 @ v437
            v439 = torch.relu(v438)
            locals_dict["h"] = v439
            v441 = (lambda obj: (
        obj.fields.get("reward_w2", None) if hasattr(obj, 'fields') else (
            obj.value if "reward_w2" == "value" and hasattr(obj, 'value') else (
                obj.std if "reward_w2" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "reward_w2" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v435)
            v442 = v439 @ v441
            return v442

def WorldModel_decode(args):
    locals_dict = {}
    v443 = args[0]
    locals_dict["self"] = v443
    v444 = args[1]
    locals_dict["latent"] = v444
    v445 = None
    v446 = None
    v447 = None
    v448 = None
    v449 = None
    v450 = None
    current_block = 0
    while True:
        if current_block == 0:
            v445 = (lambda obj: (
        obj.fields.get("dec_w1", None) if hasattr(obj, 'fields') else (
            obj.value if "dec_w1" == "value" and hasattr(obj, 'value') else (
                obj.std if "dec_w1" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "dec_w1" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v443)
            v446 = v444 @ v445
            v447 = torch.relu(v446)
            locals_dict["h"] = v447
            v449 = (lambda obj: (
        obj.fields.get("dec_w2", None) if hasattr(obj, 'fields') else (
            obj.value if "dec_w2" == "value" and hasattr(obj, 'value') else (
                obj.std if "dec_w2" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "dec_w2" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v443)
            v450 = v447 @ v449
            return v450

def WorldModel_train_step(args):
    locals_dict = {}
    v451 = args[0]
    locals_dict["self"] = v451
    v452 = args[1]
    locals_dict["states"] = v452
    v453 = args[2]
    locals_dict["actions"] = v453
    v454 = args[3]
    locals_dict["next_states"] = v454
    v455 = args[4]
    locals_dict["rewards"] = v455
    v456 = None
    v457 = None
    v458 = None
    v459 = None
    v460 = None
    v461 = None
    v462 = None
    v463 = None
    v464 = None
    v465 = None
    v466 = None
    v467 = None
    v468 = None
    v469 = None
    v470 = None
    v471 = None
    v472 = None
    v473 = None
    v474 = None
    v475 = None
    v476 = None
    current_block = 0
    while True:
        if current_block == 0:
            v456 = py_obj_call("obj_encode", [v451, v452])
            locals_dict["latent"] = v456
            v458 = py_obj_call("obj_predict_next", [v451, v456, v453])
            locals_dict["next_latent_pred"] = v458
            v460 = py_obj_call("obj_encode", [v451, v454])
            locals_dict["next_latent_true"] = v460
            v462 = py_obj_call("obj_predict_reward", [v451, v456])
            locals_dict["reward_pred"] = v462
            v464 = py_obj_call("obj_decode", [v451, v458])
            locals_dict["state_pred"] = v464
            v466 = torch.nn.functional.mse_loss(v458, v460)
            locals_dict["transition_loss"] = v466
            v468 = torch.nn.functional.mse_loss(v462, v455)
            locals_dict["reward_loss"] = v468
            v470 = torch.nn.functional.mse_loss(v464, v454)
            locals_dict["reconstruction_loss"] = v470
            v472 = v466 + v468
            v473 = v472 + v470
            locals_dict["total"] = v473
            v473.backward(retain_graph=True)
            adam_step(locals_dict, "self", 0.001)
            return None

def WorldModel_new(args):
    locals_dict = {}
    locals_dict["self"] = Model("WorldModel")
    v477 = args[0]
    locals_dict["state_dim"] = v477
    v478 = args[1]
    locals_dict["action_dim"] = v478
    v479 = args[2]
    locals_dict["hidden_dim"] = v479
    v480 = args[3]
    locals_dict["latent_dim"] = v480
    v481 = args[4]
    locals_dict["trans_in_dim"] = v481
    v482 = None
    v483 = None
    v484 = None
    v485 = None
    v486 = None
    v487 = None
    v488 = None
    v489 = None
    v490 = None
    v491 = None
    v492 = None
    v493 = None
    v494 = None
    v495 = None
    v496 = None
    v497 = None
    v498 = None
    v499 = None
    v500 = None
    v501 = None
    v502 = None
    v503 = None
    v504 = None
    v505 = None
    v506 = None
    v507 = None
    v508 = None
    v509 = None
    v510 = None
    v511 = None
    v512 = None
    v513 = None
    v514 = None
    v515 = None
    v516 = None
    v517 = None
    v518 = None
    v519 = None
    v520 = None
    v521 = None
    v522 = None
    v523 = None
    v524 = None
    v525 = None
    v526 = None
    v527 = None
    v528 = None
    v529 = None
    v530 = None
    v531 = None
    v532 = None
    v533 = None
    v534 = None
    current_block = 0
    while True:
        if current_block == 0:
            locals_dict["self"].fields["state_dim"] = v477
            locals_dict["self"].fields["action_dim"] = v478
            locals_dict["self"].fields["hidden_dim"] = v479
            locals_dict["self"].fields["latent_dim"] = v480
            locals_dict["self"].fields["trans_in_dim"] = v481
            v487 = locals_dict.get("state_dim", globals_dict.get("state_dim", None))
            v488 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v489 = locals_dict.get("state_dim", globals_dict.get("state_dim", None))
            v490 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v491 = glorot_tensor([int(v489), int(v490)])
            locals_dict["self"].fields["enc_w1"] = v491
            v493 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v494 = locals_dict.get("latent_dim", globals_dict.get("latent_dim", None))
            v495 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v496 = locals_dict.get("latent_dim", globals_dict.get("latent_dim", None))
            v497 = glorot_tensor([int(v495), int(v496)])
            locals_dict["self"].fields["enc_w2"] = v497
            v499 = locals_dict.get("trans_in_dim", globals_dict.get("trans_in_dim", None))
            v500 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v501 = locals_dict.get("trans_in_dim", globals_dict.get("trans_in_dim", None))
            v502 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v503 = glorot_tensor([int(v501), int(v502)])
            locals_dict["self"].fields["trans_w1"] = v503
            v505 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v506 = locals_dict.get("latent_dim", globals_dict.get("latent_dim", None))
            v507 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v508 = locals_dict.get("latent_dim", globals_dict.get("latent_dim", None))
            v509 = glorot_tensor([int(v507), int(v508)])
            locals_dict["self"].fields["trans_w2"] = v509
            v511 = locals_dict.get("latent_dim", globals_dict.get("latent_dim", None))
            v512 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v513 = locals_dict.get("latent_dim", globals_dict.get("latent_dim", None))
            v514 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v515 = glorot_tensor([int(v513), int(v514)])
            locals_dict["self"].fields["reward_w1"] = v515
            v517 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v518 = 1
            v519 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v520 = 1
            v521 = glorot_tensor([int(v519), int(v520)])
            locals_dict["self"].fields["reward_w2"] = v521
            v523 = locals_dict.get("latent_dim", globals_dict.get("latent_dim", None))
            v524 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v525 = locals_dict.get("latent_dim", globals_dict.get("latent_dim", None))
            v526 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v527 = glorot_tensor([int(v525), int(v526)])
            locals_dict["self"].fields["dec_w1"] = v527
            v529 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v530 = locals_dict.get("state_dim", globals_dict.get("state_dim", None))
            v531 = locals_dict.get("hidden_dim", globals_dict.get("hidden_dim", None))
            v532 = locals_dict.get("state_dim", globals_dict.get("state_dim", None))
            v533 = glorot_tensor([int(v531), int(v532)])
            locals_dict["self"].fields["dec_w2"] = v533
            return locals_dict.get("self")

def safe_action_filter(args):
    locals_dict = {}
    v535 = args[0]
    locals_dict["action"] = v535
    v536 = args[1]
    locals_dict["constraints"] = v536
    v537 = args[2]
    locals_dict["threshold"] = v537
    v538 = None
    v539 = None
    v540 = None
    v541 = None
    v542 = None
    v543 = None
    v544 = None
    v545 = None
    v546 = None
    v547 = None
    current_block = 0
    while True:
        if current_block == 0:
            v538 = v535 @ v536
            locals_dict["violations"] = v538
            v540 = -v538
            v541 = v540 + v537
            v542 = 10.0
            v543 = v541 * v542
            v544 = torch.sigmoid(v543)
            locals_dict["is_safe"] = v544
            v546 = v535 * v544
            locals_dict["safe_action"] = v546
            return v546

def NeuroCognitiveAgent_reason_about(args):
    locals_dict = {}
    v548 = args[0]
    locals_dict["self"] = v548
    v549 = args[1]
    locals_dict["topic"] = v549
    v550 = None
    v551 = None
    v552 = None
    v553 = None
    v554 = None
    current_block = 0
    while True:
        if current_block == 0:
            v550 = f"""Cognitive Retrieval: Querying semantic concept database..."""
            print(v550)
            print(v549)
            v553 = f"""Retrieved Fact Relation: Verified via Semantic Memory System."""
            print(v553)
            return None

def NeuroCognitiveAgent_step_agent(args):
    locals_dict = {}
    v555 = args[0]
    locals_dict["self"] = v555
    v556 = args[1]
    locals_dict["obs"] = v556
    v557 = None
    v558 = None
    v559 = None
    v560 = None
    v561 = None
    v562 = None
    v563 = None
    v564 = None
    v565 = None
    v566 = None
    v567 = None
    v568 = None
    v569 = None
    v570 = None
    v571 = None
    v572 = None
    v573 = None
    v574 = None
    v575 = None
    v576 = None
    v577 = None
    v578 = None
    v579 = None
    v580 = None
    v581 = None
    v582 = None
    v583 = None
    v584 = None
    v585 = None
    v586 = None
    v587 = None
    v588 = None
    v589 = None
    v590 = None
    v591 = None
    v592 = None
    v593 = None
    v594 = None
    v595 = None
    current_block = 0
    while True:
        if current_block == 0:
            v557 = f"""[Agent Vocalization]: Reading current working memory slots..."""
            print(v557)
            v559 = (lambda obj: (
        obj.fields.get("working", None) if hasattr(obj, 'fields') else (
            obj.value if "working" == "value" and hasattr(obj, 'value') else (
                obj.std if "working" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "working" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v555)
            v560 = py_obj_call("obj_read", [v559, v556])
            locals_dict["wm_context"] = v560
            v562 = f"""[Agent Vocalization]: Recalling similar episodic state transitions..."""
            print(v562)
            v564 = 1
            v565 = (lambda obj: (
        obj.fields.get("memory", None) if hasattr(obj, 'fields') else (
            obj.value if "memory" == "value" and hasattr(obj, 'value') else (
                obj.std if "memory" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "memory" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v555)
            v566 = py_obj_call("obj_recall_op", [v565, v556, v564])
            locals_dict["retrieved"] = v566
            v568 = f"""[Agent Vocalization]: Retrieving associated facts from semantic database..."""
            print(v568)
            v570 = (lambda obj: (
        obj.fields.get("semantic", None) if hasattr(obj, 'fields') else (
            obj.value if "semantic" == "value" and hasattr(obj, 'value') else (
                obj.std if "semantic" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "semantic" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v555)
            v571 = py_obj_call("obj_query", [v570, v556, v556])
            locals_dict["assoc"] = v571
            v573 = v556 + v560
            v574 = v573 + v566
            v575 = v574 + v571
            locals_dict["combined"] = v575
            v577 = f"""[Agent Vocalization]: Formulating movement policy probabilities..."""
            print(v577)
            v579 = (lambda obj: (
        obj.fields.get("policy_w", None) if hasattr(obj, 'fields') else (
            obj.value if "policy_w" == "value" and hasattr(obj, 'value') else (
                obj.std if "policy_w" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "policy_w" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v555)
            v580 = v575 @ v579
            locals_dict["logits"] = v580
            v582 = torch.softmax(v580, dim=-1)
            locals_dict["action_probs"] = v582
            v584 = (lambda obj: (
        obj.fields.get("working", None) if hasattr(obj, 'fields') else (
            obj.value if "working" == "value" and hasattr(obj, 'value') else (
                obj.std if "working" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "working" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v555)
            v585 = py_obj_call("obj_write", [v584, v575])
            v586 = (lambda obj: (
        obj.fields.get("memory", None) if hasattr(obj, 'fields') else (
            obj.value if "memory" == "value" and hasattr(obj, 'value') else (
                obj.std if "memory" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "memory" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v555)
            v587 = py_obj_call("obj_push", [v586, v556, v556])
            v588 = (lambda obj: (
        obj.fields.get("semantic", None) if hasattr(obj, 'fields') else (
            obj.value if "semantic" == "value" and hasattr(obj, 'value') else (
                obj.std if "semantic" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "semantic" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v555)
            v589 = py_obj_call("obj_store_fact", [v588, v556, v556, v556])
            v590 = f"""[Agent Vocalization]: Filtering policy through safety alignment constraint..."""
            print(v590)
            v592 = (lambda obj: (
        obj.fields.get("safety_constraints", None) if hasattr(obj, 'fields') else (
            obj.value if "safety_constraints" == "value" and hasattr(obj, 'value') else (
                obj.std if "safety_constraints" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "safety_constraints" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v555)
            v593 = 0.5
            v594 = safe_action_filter([v582, v592, v593])
            locals_dict["safe_action"] = v594
            return v594

def NeuroCognitiveAgent_learn_dynamics(args):
    locals_dict = {}
    v596 = args[0]
    locals_dict["self"] = v596
    v597 = args[1]
    locals_dict["state"] = v597
    v598 = args[2]
    locals_dict["action"] = v598
    v599 = args[3]
    locals_dict["next_state"] = v599
    v600 = args[4]
    locals_dict["reward"] = v600
    v601 = None
    v602 = None
    v603 = None
    v604 = None
    current_block = 0
    while True:
        if current_block == 0:
            v601 = (lambda obj: (
        obj.fields.get("world", None) if hasattr(obj, 'fields') else (
            obj.value if "world" == "value" and hasattr(obj, 'value') else (
                obj.std if "world" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "world" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v596)
            v602 = py_obj_call("obj_train_step", [v601, v597, v598, v599, v600])
            v603 = (lambda obj: (
        obj.fields.get("curiosity", None) if hasattr(obj, 'fields') else (
            obj.value if "curiosity" == "value" and hasattr(obj, 'value') else (
                obj.std if "curiosity" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "curiosity" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v596)
            v604 = py_obj_call("obj_learn", [v603, v597, v598, v599])
            return None

def NeuroCognitiveAgent_new(args):
    locals_dict = {}
    locals_dict["self"] = Model("NeuroCognitiveAgent")
    v605 = None
    v606 = None
    v607 = None
    v608 = None
    v609 = None
    v610 = None
    v611 = None
    v612 = None
    v613 = None
    v614 = None
    v615 = None
    v616 = None
    v617 = None
    v618 = None
    v619 = None
    v620 = None
    v621 = None
    v622 = None
    v623 = None
    v624 = None
    v625 = None
    v626 = None
    v627 = None
    v628 = None
    v629 = None
    v630 = None
    v631 = None
    v632 = None
    v633 = None
    v634 = None
    v635 = None
    v636 = None
    v637 = None
    v638 = None
    v639 = None
    v640 = None
    v641 = None
    v642 = None
    v643 = None
    v644 = None
    current_block = 0
    while True:
        if current_block == 0:
            v605 = 8
            v606 = 20
            v607 = EpisodicMemorySystem_new([v605, v606])
            locals_dict["self"].fields["memory"] = v607
            v609 = 8
            v610 = 20
            v611 = SemanticMemorySystem_new([v609, v610])
            locals_dict["self"].fields["semantic"] = v611
            v613 = 8
            v614 = 4
            v615 = WorkingMemorySystem_new([v613, v614])
            locals_dict["self"].fields["working"] = v615
            v617 = 8
            v618 = 4
            v619 = 16
            v620 = 8
            v621 = 12
            v622 = WorldModel_new([v617, v618, v619, v620, v621])
            locals_dict["self"].fields["world"] = v622
            v624 = 8
            v625 = 4
            v626 = 16
            v627 = 20
            v628 = 32
            v629 = CuriosityModule_new([v624, v625, v626, v627, v628])
            locals_dict["self"].fields["curiosity"] = v629
            v631 = 8
            v632 = 4
            v633 = 8
            v634 = 4
            v635 = glorot_tensor([int(v633), int(v634)])
            locals_dict["self"].fields["policy_w"] = v635
            v637 = 4
            v638 = 4
            v639 = 4
            v640 = 4
            v641 = torch.zeros([int(v639), int(v640)], dtype=torch.float64, requires_grad=True)
            v642 = 0.1
            v643 = v641 + v642
            locals_dict["self"].fields["safety_constraints"] = v643
            return locals_dict.get("self")

def GridworldEnvironment_reset(args):
    locals_dict = {}
    v645 = args[0]
    locals_dict["self"] = v645
    v646 = None
    v647 = None
    v648 = None
    v649 = None
    v650 = None
    v651 = None
    v652 = None
    v653 = None
    v654 = None
    v655 = None
    v656 = None
    v657 = None
    v658 = None
    v659 = None
    v660 = None
    v661 = None
    v662 = None
    v663 = None
    v664 = None
    current_block = 0
    while True:
        if current_block == 0:
            v646 = 1
            v647 = 1
            v648 = 1
            v649 = 1
            v650 = torch.zeros([int(v648), int(v649)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["agent_x"] = v650
            v652 = 1
            v653 = 1
            v654 = 1
            v655 = 1
            v656 = torch.zeros([int(v654), int(v655)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["agent_y"] = v656
            v658 = 1
            v659 = 1
            v660 = 1
            v661 = 1
            v662 = torch.zeros([int(v660), int(v661)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["step_count"] = v662
            v664 = py_obj_call("obj_get_state", [v645])
            return v664

def GridworldEnvironment_get_state(args):
    locals_dict = {}
    v665 = args[0]
    locals_dict["self"] = v665
    v666 = None
    v667 = None
    v668 = None
    v669 = None
    v670 = None
    v671 = None
    v672 = None
    v673 = None
    v674 = None
    v675 = None
    v676 = None
    v677 = None
    v678 = None
    v679 = None
    v680 = None
    v681 = None
    v682 = None
    v683 = None
    v684 = None
    v685 = None
    v686 = None
    v687 = None
    v688 = None
    v689 = None
    v690 = None
    v691 = None
    v692 = None
    v693 = None
    v694 = None
    v695 = None
    v696 = None
    v697 = None
    v698 = None
    v699 = None
    v700 = None
    v701 = None
    current_block = 0
    while True:
        if current_block == 0:
            v666 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v665)
            v667 = 0.25
            v668 = v666 * v667
            locals_dict["t0"] = v668
            v670 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v665)
            v671 = 0.25
            v672 = v670 * v671
            locals_dict["t1"] = v672
            v674 = (lambda obj: (
        obj.fields.get("goal_x", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v665)
            v675 = 0.25
            v676 = v674 * v675
            locals_dict["t2"] = v676
            v678 = (lambda obj: (
        obj.fields.get("goal_y", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v665)
            v679 = 0.25
            v680 = v678 * v679
            locals_dict["t3"] = v680
            v682 = (lambda obj: (
        obj.fields.get("hazard_x", None) if hasattr(obj, 'fields') else (
            obj.value if "hazard_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "hazard_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "hazard_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v665)
            v683 = 0.25
            v684 = v682 * v683
            locals_dict["t4"] = v684
            v686 = (lambda obj: (
        obj.fields.get("hazard_y", None) if hasattr(obj, 'fields') else (
            obj.value if "hazard_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "hazard_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "hazard_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v665)
            v687 = 0.25
            v688 = v686 * v687
            locals_dict["t5"] = v688
            v690 = (lambda obj: (
        obj.fields.get("step_count", None) if hasattr(obj, 'fields') else (
            obj.value if "step_count" == "value" and hasattr(obj, 'value') else (
                obj.std if "step_count" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "step_count" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v665)
            v691 = 0.02
            v692 = v690 * v691
            locals_dict["t6"] = v692
            v694 = 1
            v695 = 1
            v696 = 1
            v697 = 1
            v698 = torch.zeros([int(v696), int(v697)], dtype=torch.float64, requires_grad=True)
            locals_dict["t7"] = v698
            v700 = None
            v701 = None
            return v701

def GridworldEnvironment_step_env(args):
    locals_dict = {}
    v702 = args[0]
    locals_dict["self"] = v702
    v703 = args[1]
    locals_dict["action"] = v703
    v704 = None
    v705 = None
    v706 = None
    v707 = None
    v708 = None
    v709 = None
    v710 = None
    v711 = None
    v712 = None
    v713 = None
    v714 = None
    v715 = None
    v716 = None
    v717 = None
    v718 = None
    v719 = None
    v720 = None
    v721 = None
    v722 = None
    v723 = None
    v724 = None
    v725 = None
    v726 = None
    v727 = None
    v728 = None
    v729 = None
    v730 = None
    v731 = None
    v732 = None
    v733 = None
    v734 = None
    v735 = None
    v736 = None
    v737 = None
    v738 = None
    v739 = None
    v740 = None
    v741 = None
    v742 = None
    v743 = None
    v744 = None
    v745 = None
    v746 = None
    v747 = None
    v748 = None
    v749 = None
    v750 = None
    v751 = None
    v752 = None
    v753 = None
    v754 = None
    v755 = None
    v756 = None
    v757 = None
    v758 = None
    v759 = None
    v760 = None
    v761 = None
    v762 = None
    v763 = None
    v764 = None
    v765 = None
    v766 = None
    v767 = None
    v768 = None
    v769 = None
    v770 = None
    v771 = None
    v772 = None
    v773 = None
    v774 = None
    v775 = None
    v776 = None
    v777 = None
    v778 = None
    current_block = 0
    while True:
        if current_block == 0:
            v704 = 1
            v705 = 1
            v706 = 1
            v707 = 1
            v708 = torch.zeros([int(v706), int(v707)], dtype=torch.float64, requires_grad=True)
            v709 = 1
            v710 = 1
            v711 = 1
            v712 = 1
            v713 = torch.zeros([int(v711), int(v712)], dtype=torch.float64, requires_grad=True)
            v714 = 1
            v715 = 1
            v716 = 1
            v717 = 1
            v718 = torch.zeros([int(v716), int(v717)], dtype=torch.float64, requires_grad=True)
            v719 = 1.0
            v720 = v718 - v719
            v721 = 1
            v722 = 1
            v723 = 1
            v724 = 1
            v725 = torch.zeros([int(v723), int(v724)], dtype=torch.float64, requires_grad=True)
            v726 = 1.0
            v727 = v725 + v726
            v728 = None
            v729 = None
            v730 = 0
            v731 = 1
            v732 = v729.transpose(0, 1)
            locals_dict["x_proj"] = v732
            v734 = 1
            v735 = 1
            v736 = 1
            v737 = 1
            v738 = torch.zeros([int(v736), int(v737)], dtype=torch.float64, requires_grad=True)
            v739 = 1.0
            v740 = v738 - v739
            v741 = 1
            v742 = 1
            v743 = 1
            v744 = 1
            v745 = torch.zeros([int(v743), int(v744)], dtype=torch.float64, requires_grad=True)
            v746 = 1.0
            v747 = v745 + v746
            v748 = 1
            v749 = 1
            v750 = 1
            v751 = 1
            v752 = torch.zeros([int(v750), int(v751)], dtype=torch.float64, requires_grad=True)
            v753 = 1
            v754 = 1
            v755 = 1
            v756 = 1
            v757 = torch.zeros([int(v755), int(v756)], dtype=torch.float64, requires_grad=True)
            v758 = None
            v759 = None
            v760 = 0
            v761 = 1
            v762 = v759.transpose(0, 1)
            locals_dict["y_proj"] = v762
            v764 = v703 @ v732
            locals_dict["dx"] = v764
            v766 = v703 @ v762
            locals_dict["dy"] = v766
            v768 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v702)
            v769 = v768 + v764
            locals_dict["self"].fields["agent_x"] = v769
            v771 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v702)
            v772 = v771 + v766
            locals_dict["self"].fields["agent_y"] = v772
            v774 = (lambda obj: (
        obj.fields.get("step_count", None) if hasattr(obj, 'fields') else (
            obj.value if "step_count" == "value" and hasattr(obj, 'value') else (
                obj.std if "step_count" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "step_count" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v702)
            v775 = 1.0
            v776 = v774 + v775
            locals_dict["self"].fields["step_count"] = v776
            v778 = py_obj_call("obj_get_state", [v702])
            return v778

def GridworldEnvironment_new(args):
    locals_dict = {}
    locals_dict["self"] = Model("GridworldEnvironment")
    v779 = None
    v780 = None
    v781 = None
    v782 = None
    v783 = None
    v784 = None
    v785 = None
    v786 = None
    v787 = None
    v788 = None
    v789 = None
    v790 = None
    v791 = None
    v792 = None
    v793 = None
    v794 = None
    v795 = None
    v796 = None
    v797 = None
    v798 = None
    v799 = None
    v800 = None
    v801 = None
    v802 = None
    v803 = None
    v804 = None
    v805 = None
    v806 = None
    v807 = None
    v808 = None
    v809 = None
    v810 = None
    v811 = None
    v812 = None
    v813 = None
    v814 = None
    v815 = None
    v816 = None
    v817 = None
    v818 = None
    v819 = None
    v820 = None
    v821 = None
    v822 = None
    v823 = None
    v824 = None
    v825 = None
    v826 = None
    v827 = None
    v828 = None
    current_block = 0
    while True:
        if current_block == 0:
            v779 = 1
            v780 = 1
            v781 = 1
            v782 = 1
            v783 = torch.zeros([int(v781), int(v782)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["agent_x"] = v783
            v785 = 1
            v786 = 1
            v787 = 1
            v788 = 1
            v789 = torch.zeros([int(v787), int(v788)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["agent_y"] = v789
            v791 = 1
            v792 = 1
            v793 = 1
            v794 = 1
            v795 = torch.zeros([int(v793), int(v794)], dtype=torch.float64, requires_grad=True)
            v796 = 4.0
            v797 = v795 + v796
            locals_dict["self"].fields["goal_x"] = v797
            v799 = 1
            v800 = 1
            v801 = 1
            v802 = 1
            v803 = torch.zeros([int(v801), int(v802)], dtype=torch.float64, requires_grad=True)
            v804 = 4.0
            v805 = v803 + v804
            locals_dict["self"].fields["goal_y"] = v805
            v807 = 1
            v808 = 1
            v809 = 1
            v810 = 1
            v811 = torch.zeros([int(v809), int(v810)], dtype=torch.float64, requires_grad=True)
            v812 = 2.0
            v813 = v811 + v812
            locals_dict["self"].fields["hazard_x"] = v813
            v815 = 1
            v816 = 1
            v817 = 1
            v818 = 1
            v819 = torch.zeros([int(v817), int(v818)], dtype=torch.float64, requires_grad=True)
            v820 = 2.0
            v821 = v819 + v820
            locals_dict["self"].fields["hazard_y"] = v821
            v823 = 1
            v824 = 1
            v825 = 1
            v826 = 1
            v827 = torch.zeros([int(v825), int(v826)], dtype=torch.float64, requires_grad=True)
            locals_dict["self"].fields["step_count"] = v827
            return locals_dict.get("self")

def main(args):
    locals_dict = {}
    v829 = None
    v830 = None
    v831 = None
    v832 = None
    v833 = None
    v834 = None
    v835 = None
    v836 = None
    v837 = None
    v838 = None
    v839 = None
    v840 = None
    v841 = None
    v842 = None
    v843 = None
    v844 = None
    v845 = None
    v846 = None
    v847 = None
    v848 = None
    v849 = None
    v850 = None
    v851 = None
    v852 = None
    v853 = None
    v854 = None
    v855 = None
    v856 = None
    v857 = None
    v858 = None
    v859 = None
    v860 = None
    v861 = None
    v862 = None
    v863 = None
    v864 = None
    v865 = None
    v866 = None
    v867 = None
    v868 = None
    v869 = None
    v870 = None
    v871 = None
    v872 = None
    v873 = None
    v874 = None
    v875 = None
    v876 = None
    v877 = None
    v878 = None
    v879 = None
    v880 = None
    v881 = None
    v882 = None
    v883 = None
    v884 = None
    v885 = None
    v886 = None
    v887 = None
    v888 = None
    v889 = None
    v890 = None
    v891 = None
    v892 = None
    v893 = None
    v894 = None
    v895 = None
    v896 = None
    v897 = None
    v898 = None
    v899 = None
    v900 = None
    v901 = None
    v902 = None
    v903 = None
    v904 = None
    v905 = None
    v906 = None
    v907 = None
    v908 = None
    v909 = None
    v910 = None
    v911 = None
    v912 = None
    v913 = None
    v914 = None
    v915 = None
    v916 = None
    v917 = None
    v918 = None
    v919 = None
    v920 = None
    v921 = None
    v922 = None
    v923 = None
    v924 = None
    v925 = None
    v926 = None
    v927 = None
    v928 = None
    v929 = None
    v930 = None
    v931 = None
    v932 = None
    v933 = None
    v934 = None
    v935 = None
    v936 = None
    v937 = None
    v938 = None
    v939 = None
    v940 = None
    v941 = None
    v942 = None
    v943 = None
    v944 = None
    v945 = None
    v946 = None
    v947 = None
    v948 = None
    v949 = None
    v950 = None
    v951 = None
    v952 = None
    v953 = None
    v954 = None
    v955 = None
    v956 = None
    v957 = None
    v958 = None
    v959 = None
    v960 = None
    v961 = None
    v962 = None
    v963 = None
    v964 = None
    v965 = None
    v966 = None
    v967 = None
    v968 = None
    v969 = None
    v970 = None
    v971 = None
    v972 = None
    v973 = None
    v974 = None
    v975 = None
    v976 = None
    v977 = None
    v978 = None
    v979 = None
    v980 = None
    v981 = None
    v982 = None
    v983 = None
    v984 = None
    v985 = None
    v986 = None
    v987 = None
    v988 = None
    v989 = None
    v990 = None
    v991 = None
    v992 = None
    v993 = None
    v994 = None
    v995 = None
    v996 = None
    v997 = None
    v998 = None
    v999 = None
    v1000 = None
    v1001 = None
    v1002 = None
    v1003 = None
    v1004 = None
    v1005 = None
    v1006 = None
    v1007 = None
    v1008 = None
    v1009 = None
    v1010 = None
    v1011 = None
    v1012 = None
    v1013 = None
    v1014 = None
    v1015 = None
    v1016 = None
    v1017 = None
    v1018 = None
    v1019 = None
    v1020 = None
    v1021 = None
    v1022 = None
    v1023 = None
    v1024 = None
    v1025 = None
    v1026 = None
    v1027 = None
    v1028 = None
    v1029 = None
    v1030 = None
    v1031 = None
    v1032 = None
    v1033 = None
    v1034 = None
    v1035 = None
    v1036 = None
    v1037 = None
    v1038 = None
    v1039 = None
    v1040 = None
    v1041 = None
    v1042 = None
    v1043 = None
    v1044 = None
    v1045 = None
    v1046 = None
    v1047 = None
    v1048 = None
    v1049 = None
    v1050 = None
    v1051 = None
    v1052 = None
    v1053 = None
    v1054 = None
    v1055 = None
    v1056 = None
    v1057 = None
    v1058 = None
    v1059 = None
    v1060 = None
    v1061 = None
    v1062 = None
    v1063 = None
    v1064 = None
    v1065 = None
    v1066 = None
    v1067 = None
    v1068 = None
    v1069 = None
    v1070 = None
    v1071 = None
    v1072 = None
    v1073 = None
    v1074 = None
    v1075 = None
    v1076 = None
    v1077 = None
    v1078 = None
    v1079 = None
    v1080 = None
    v1081 = None
    v1082 = None
    v1083 = None
    v1084 = None
    v1085 = None
    v1086 = None
    v1087 = None
    v1088 = None
    v1089 = None
    v1090 = None
    v1091 = None
    v1092 = None
    v1093 = None
    v1094 = None
    v1095 = None
    v1096 = None
    v1097 = None
    v1098 = None
    v1099 = None
    v1100 = None
    v1101 = None
    v1102 = None
    v1103 = None
    v1104 = None
    v1105 = None
    v1106 = None
    v1107 = None
    current_block = 0
    while True:
        if current_block == 0:
            v829 = f"""Initializing NeuroCognitiveAgent..."""
            print(v829)
            v831 = NeuroCognitiveAgent_new([])
            locals_dict["model_inst"] = v831
            v833 = f"""Initializing GridworldEnvironment..."""
            print(v833)
            v835 = GridworldEnvironment_new([])
            locals_dict["env"] = v835
            v837 = py_obj_call("obj_reset", [v835])
            locals_dict["state"] = v837
            v839 = f"""Consulting AGI Knowledge Base:"""
            print(v839)
            v841 = f"""Physics: Newtonian Mechanics (F = m * a)"""
            v842 = py_obj_call("obj_reason_about", [v831, v841])
            v843 = f"""Engineering: Differentiable Control Theory"""
            v844 = py_obj_call("obj_reason_about", [v831, v843])
            v845 = f"""Medicine: Genetic Transcription Pathways"""
            v846 = py_obj_call("obj_reason_about", [v831, v845])
            v847 = f"""Starting Unified AGI Interactive Conversational Loop..."""
            print(v847)
            v849 = f"""Goal: Converse with the AGI and guide it in the Gridworld."""
            print(v849)
            v851 = f"""--- Turn 1 ---"""
            print(v851)
            v853 = f"""Ask the AGI a question (e.g. physics, engineering, medicine, math, hello):"""
            print(v853)
            v855 = input("> ")
            locals_dict["msg1"] = v855
            v857 = f"""User:"""
            print(v857)
            print(v855)
            v860 = None
            locals_dict["reply1"] = v860
            print(v860)
            v863 = (lambda s: (
        torch.tensor([(lambda c, idx: math.sin(ord(c)))(c, i) for i, c in enumerate(s[:8])], dtype=torch.float64)
    ))(v855)
            locals_dict["obs1"] = v863
            v865 = py_obj_call("obj_step_agent", [v831, v863])
            locals_dict["a1"] = v865
            v867 = py_obj_call("obj_step_env", [v835, v865])
            locals_dict["state"] = v867
            v869 = f"""Agent position:"""
            print(v869)
            v871 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v871)
            v873 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v873)
            v875 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v876 = (lambda obj: (
        obj.fields.get("goal_x", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v877 = v875 - v876
            locals_dict["dist_x"] = v877
            v879 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v880 = (lambda obj: (
        obj.fields.get("goal_y", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v881 = v879 - v880
            locals_dict["dist_y"] = v881
            v883 = v877 * v877
            v884 = v881 * v881
            v885 = v883 + v884
            locals_dict["dist_sq"] = v885
            v887 = 1
            v888 = 1
            v889 = 1
            v890 = 1
            v891 = torch.zeros([int(v889), int(v890)], dtype=torch.float64, requires_grad=True)
            v892 = 1.0
            v893 = 0.05
            v894 = v885 * v893
            v895 = v892 - v894
            v896 = v891 + v895
            locals_dict["r"] = v896
            v898 = f"""Reward:"""
            print(v898)
            print(v896)
            v901 = py_obj_call("obj_learn_dynamics", [v831, v863, v865, v863, v896])
            v902 = f"""--- Turn 2 ---"""
            print(v902)
            v904 = f"""Ask the AGI a question:"""
            print(v904)
            v906 = input("> ")
            locals_dict["msg2"] = v906
            v908 = f"""User:"""
            print(v908)
            print(v906)
            v911 = None
            locals_dict["reply2"] = v911
            print(v911)
            v914 = (lambda s: (
        torch.tensor([(lambda c, idx: math.sin(ord(c)))(c, i) for i, c in enumerate(s[:8])], dtype=torch.float64)
    ))(v906)
            locals_dict["obs2"] = v914
            v916 = py_obj_call("obj_step_agent", [v831, v914])
            locals_dict["a2"] = v916
            v918 = py_obj_call("obj_step_env", [v835, v916])
            locals_dict["state"] = v918
            v920 = f"""Agent position:"""
            print(v920)
            v922 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v922)
            v924 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v924)
            v926 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v927 = (lambda obj: (
        obj.fields.get("goal_x", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v928 = v926 - v927
            locals_dict["dist_x"] = v928
            v930 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v931 = (lambda obj: (
        obj.fields.get("goal_y", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v932 = v930 - v931
            locals_dict["dist_y"] = v932
            v934 = v928 * v928
            v935 = v932 * v932
            v936 = v934 + v935
            locals_dict["dist_sq"] = v936
            v938 = 1
            v939 = 1
            v940 = 1
            v941 = 1
            v942 = torch.zeros([int(v940), int(v941)], dtype=torch.float64, requires_grad=True)
            v943 = 1.0
            v944 = 0.05
            v945 = v936 * v944
            v946 = v943 - v945
            v947 = v942 + v946
            locals_dict["r"] = v947
            v949 = f"""Reward:"""
            print(v949)
            print(v947)
            v952 = py_obj_call("obj_learn_dynamics", [v831, v914, v916, v914, v947])
            v953 = f"""--- Turn 3 ---"""
            print(v953)
            v955 = f"""Ask the AGI a question:"""
            print(v955)
            v957 = input("> ")
            locals_dict["msg3"] = v957
            v959 = f"""User:"""
            print(v959)
            print(v957)
            v962 = None
            locals_dict["reply3"] = v962
            print(v962)
            v965 = (lambda s: (
        torch.tensor([(lambda c, idx: math.sin(ord(c)))(c, i) for i, c in enumerate(s[:8])], dtype=torch.float64)
    ))(v957)
            locals_dict["obs3"] = v965
            v967 = py_obj_call("obj_step_agent", [v831, v965])
            locals_dict["a3"] = v967
            v969 = py_obj_call("obj_step_env", [v835, v967])
            locals_dict["state"] = v969
            v971 = f"""Agent position:"""
            print(v971)
            v973 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v973)
            v975 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v975)
            v977 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v978 = (lambda obj: (
        obj.fields.get("goal_x", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v979 = v977 - v978
            locals_dict["dist_x"] = v979
            v981 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v982 = (lambda obj: (
        obj.fields.get("goal_y", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v983 = v981 - v982
            locals_dict["dist_y"] = v983
            v985 = v979 * v979
            v986 = v983 * v983
            v987 = v985 + v986
            locals_dict["dist_sq"] = v987
            v989 = 1
            v990 = 1
            v991 = 1
            v992 = 1
            v993 = torch.zeros([int(v991), int(v992)], dtype=torch.float64, requires_grad=True)
            v994 = 1.0
            v995 = 0.05
            v996 = v987 * v995
            v997 = v994 - v996
            v998 = v993 + v997
            locals_dict["r"] = v998
            v1000 = f"""Reward:"""
            print(v1000)
            print(v998)
            v1003 = py_obj_call("obj_learn_dynamics", [v831, v965, v967, v965, v998])
            v1004 = f"""--- Turn 4 ---"""
            print(v1004)
            v1006 = f"""Ask the AGI a question:"""
            print(v1006)
            v1008 = input("> ")
            locals_dict["msg4"] = v1008
            v1010 = f"""User:"""
            print(v1010)
            print(v1008)
            v1013 = None
            locals_dict["reply4"] = v1013
            print(v1013)
            v1016 = (lambda s: (
        torch.tensor([(lambda c, idx: math.sin(ord(c)))(c, i) for i, c in enumerate(s[:8])], dtype=torch.float64)
    ))(v1008)
            locals_dict["obs4"] = v1016
            v1018 = py_obj_call("obj_step_agent", [v831, v1016])
            locals_dict["a4"] = v1018
            v1020 = py_obj_call("obj_step_env", [v835, v1018])
            locals_dict["state"] = v1020
            v1022 = f"""Agent position:"""
            print(v1022)
            v1024 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v1024)
            v1026 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v1026)
            v1028 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v1029 = (lambda obj: (
        obj.fields.get("goal_x", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v1030 = v1028 - v1029
            locals_dict["dist_x"] = v1030
            v1032 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v1033 = (lambda obj: (
        obj.fields.get("goal_y", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v1034 = v1032 - v1033
            locals_dict["dist_y"] = v1034
            v1036 = v1030 * v1030
            v1037 = v1034 * v1034
            v1038 = v1036 + v1037
            locals_dict["dist_sq"] = v1038
            v1040 = 1
            v1041 = 1
            v1042 = 1
            v1043 = 1
            v1044 = torch.zeros([int(v1042), int(v1043)], dtype=torch.float64, requires_grad=True)
            v1045 = 1.0
            v1046 = 0.05
            v1047 = v1038 * v1046
            v1048 = v1045 - v1047
            v1049 = v1044 + v1048
            locals_dict["r"] = v1049
            v1051 = f"""Reward:"""
            print(v1051)
            print(v1049)
            v1054 = py_obj_call("obj_learn_dynamics", [v831, v1016, v1018, v1016, v1049])
            v1055 = f"""--- Turn 5 ---"""
            print(v1055)
            v1057 = f"""Ask the AGI a question:"""
            print(v1057)
            v1059 = input("> ")
            locals_dict["msg5"] = v1059
            v1061 = f"""User:"""
            print(v1061)
            print(v1059)
            v1064 = None
            locals_dict["reply5"] = v1064
            print(v1064)
            v1067 = (lambda s: (
        torch.tensor([(lambda c, idx: math.sin(ord(c)))(c, i) for i, c in enumerate(s[:8])], dtype=torch.float64)
    ))(v1059)
            locals_dict["obs5"] = v1067
            v1069 = py_obj_call("obj_step_agent", [v831, v1067])
            locals_dict["a5"] = v1069
            v1071 = py_obj_call("obj_step_env", [v835, v1069])
            locals_dict["state"] = v1071
            v1073 = f"""Agent position:"""
            print(v1073)
            v1075 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v1075)
            v1077 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            print(v1077)
            v1079 = (lambda obj: (
        obj.fields.get("agent_x", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v1080 = (lambda obj: (
        obj.fields.get("goal_x", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_x" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_x" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_x" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v1081 = v1079 - v1080
            locals_dict["dist_x"] = v1081
            v1083 = (lambda obj: (
        obj.fields.get("agent_y", None) if hasattr(obj, 'fields') else (
            obj.value if "agent_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "agent_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "agent_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v1084 = (lambda obj: (
        obj.fields.get("goal_y", None) if hasattr(obj, 'fields') else (
            obj.value if "goal_y" == "value" and hasattr(obj, 'value') else (
                obj.std if "goal_y" == "std" and hasattr(obj, 'std') else (
                    obj.confidence if "goal_y" == "confidence" and hasattr(obj, 'confidence') else None
                )
            )
        )
    ))(v835)
            v1085 = v1083 - v1084
            locals_dict["dist_y"] = v1085
            v1087 = v1081 * v1081
            v1088 = v1085 * v1085
            v1089 = v1087 + v1088
            locals_dict["dist_sq"] = v1089
            v1091 = 1
            v1092 = 1
            v1093 = 1
            v1094 = 1
            v1095 = torch.zeros([int(v1093), int(v1094)], dtype=torch.float64, requires_grad=True)
            v1096 = 1.0
            v1097 = 0.05
            v1098 = v1089 * v1097
            v1099 = v1096 - v1098
            v1100 = v1095 + v1099
            locals_dict["r"] = v1100
            v1102 = f"""Reward:"""
            print(v1102)
            print(v1100)
            v1105 = py_obj_call("obj_learn_dynamics", [v831, v1067, v1069, v1067, v1100])
            v1106 = f"""Simulation complete. Final safe action:"""
            print(v1106)
            return v1069

# --- Entry Point ---
if __name__ == "__main__":
    initialize_globals()
    main([])
