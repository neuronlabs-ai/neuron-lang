// ═══════════════════════════════════════════════════════════════
//  NEURON VS Code Extension — Language Client
//  Connects to the neuronc LSP server for real-time diagnostics.
// ═══════════════════════════════════════════════════════════════

const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let client;

function activate(context) {
    console.log('[NEURON] Extension activating...');

    // Resolve the neuronc binary path
    const config = vscode.workspace.getConfiguration('neuron');
    let compilerPath = config.get('compilerPath', 'neuronc');

    // Try common locations if not in PATH
    if (compilerPath === 'neuronc') {
        const tryPaths = [
            path.join(context.extensionPath, '..', '..', '..', 'target', 'release', 'neuronc'),
            path.join(context.extensionPath, '..', '..', '..', 'target', 'release', 'neuronc.exe'),
            path.join(context.extensionPath, '..', '..', 'target', 'release', 'neuronc'),
            path.join(context.extensionPath, '..', '..', 'target', 'release', 'neuronc.exe'),
        ];
        for (const p of tryPaths) {
            if (fs.existsSync(p)) {
                compilerPath = p;
                break;
            }
        }
    }

    // Server options: run neuronc lsp over stdio
    const serverOptions = {
        run: {
            command: compilerPath,
            args: ['lsp'],
            transport: TransportKind.stdio
        },
        debug: {
            command: compilerPath,
            args: ['lsp'],
            transport: TransportKind.stdio
        }
    };

    // Client options: activate for .nr files
    const clientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'neuron' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.nr')
        }
    };

    // Create and start the language client
    client = new LanguageClient(
        'neuron-lsp',
        'NEURON Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
    console.log('[NEURON] Language client started.');

    // Register status bar item
    const statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusItem.text = '$(beaker) NEURON';
    statusItem.tooltip = 'NEURON Language Server active';
    statusItem.show();
    context.subscriptions.push(statusItem);
}

function deactivate() {
    if (client) {
        return client.stop();
    }
}

module.exports = { activate, deactivate };
