# NEURON for Visual Studio Code

Official VS Code language support for **NEURON** — the sovereign AI-native programming language designed for differentiable programming, causal reasoning, temporal streams, and certified cognitive unlearning.

---

## Features

- **Full Syntax Highlighting**: Modern TextMate grammar covering all NEURON keywords (`fn`, `let`, `model`, `agent`, `causal`, `temporal`), types (`Tensor`, `Stream`, `Offset`, `Uncertain`, `EpisodicMemory`), cognitive operators (`update by`, `grad`, `intervene`, `do`, `forget`, `verify`), and operators (`@`, `::`, `<-`, `~`).
- **Real-Time Language Server (LSP)**:
  - Syntax and type checking on save
  - Diagnostics with red squiggly error indicators and precise line numbers
  - Hover tooltips with inferred types and documentation
  - Code completion for standard library and user definitions
- **One-Click Commands**:
  - `NEURON: Type Check Current File` (`neuron.check`)
  - `NEURON: Run Current File (VM)` (`neuron.run`)
  - `NEURON: JIT Run Current File (Native)` (`neuron.jit`)
  - `NEURON: Transpile to PyTorch Python` (`neuron.transpile`)
  - `NEURON: Restart Language Server` (`neuron.restartServer`)

---

## Installation & Setup

1. **Build `neuronc`**:
   Ensure `neuronc` is built and available on your system:
   ```bash
   cargo build --release
   ```

2. **Install Extension**:
   Copy or symlink this folder to your VS Code extensions directory:
   - **Windows**: `%USERPROFILE%\.vscode\extensions\neuron-lang-vscode`
   - **Linux / macOS**: `~/.vscode/extensions/neuron-lang-vscode`

   Or install dependencies and pack into a `.vsix`:
   ```bash
   cd editors/vscode
   npm install
   npx @vscode/vsce package
   code --install-extension neuron-lang-vscode-1.0.0.vsix
   ```

3. **Configure Path (Optional)**:
   If `neuronc` is not on your system `PATH`, configure its path in VS Code settings:
   ```json
   {
     "neuron.compilerPath": "C:/Users/ADMIN/neuron-lang/target/release/neuronc.exe"
   }
   ```

---

## License

Licensed under the MIT License.
