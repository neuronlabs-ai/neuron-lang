// ═══════════════════════════════════════════════════════════════
//  NEURON VS Code Extension — Official Language Client & Tooling
//  Connects to the neuronc LSP server for real-time diagnostics,
//  hover type info, completions, and execution commands.
// ═══════════════════════════════════════════════════════════════

const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');
const path = require('path');
const fs = require('fs');

let client;
let neuronTerminal = null;

function resolveCompilerPath(context) {
    const config = vscode.workspace.getConfiguration('neuron');
    let compilerPath = config.get('compilerPath', 'neuronc');

    if (compilerPath === 'neuronc') {
        const isWindows = process.platform === 'win32';
        const binName = isWindows ? 'neuronc.exe' : 'neuronc';

        // Check common target directories in workspace or relative paths
        const searchPaths = [
            path.join(context.extensionPath, '..', '..', 'target', 'release', binName),
            path.join(context.extensionPath, '..', '..', 'target', 'debug', binName),
            path.join(context.extensionPath, '..', '..', '..', 'target', 'release', binName),
        ];

        if (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0) {
            for (const folder of vscode.workspace.workspaceFolders) {
                searchPaths.unshift(path.join(folder.uri.fsPath, 'target', 'release', binName));
                searchPaths.unshift(path.join(folder.uri.fsPath, 'target', 'debug', binName));
                searchPaths.unshift(path.join(folder.uri.fsPath, binName));
            }
        }

        for (const candidate of searchPaths) {
            if (fs.existsSync(candidate)) {
                return candidate;
            }
        }
    }

    return compilerPath;
}

function getTerminal() {
    if (!neuronTerminal || neuronTerminal.exitStatus !== undefined) {
        neuronTerminal = vscode.window.createTerminal('NEURON');
    }
    return neuronTerminal;
}

function runInTerminal(command) {
    const term = getTerminal();
    term.show();
    term.sendText(command);
}

function getActiveFilePath() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active NEURON file open in editor.');
        return null;
    }
    if (editor.document.languageId !== 'neuron' && !editor.document.fileName.endsWith('.nr')) {
        vscode.window.showWarningMessage('The active file is not a NEURON (.nr) source file.');
    }
    return editor.document.fileName;
}

function activate(context) {
    console.log('[NEURON] Extension activating...');

    const compilerPath = resolveCompilerPath(context);

    // 1. Language Server Setup (neuronc lsp)
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

    const clientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'neuron' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.nr')
        }
    };

    client = new LanguageClient(
        'neuron-lsp',
        'NEURON Language Server',
        serverOptions,
        clientOptions
    );

    client.start().catch(err => {
        console.warn('[NEURON] Failed to start language server:', err.message);
        vscode.window.showWarningMessage(
            `NEURON LSP could not be started with '${compilerPath}'. Make sure 'neuronc' is in your PATH or configured in settings.`
        );
    });

    // 2. Status Bar Item
    const statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusItem.text = '$(beaker) NEURON';
    statusItem.tooltip = `NEURON Language Support (compiler: ${compilerPath})`;
    statusItem.show();
    context.subscriptions.push(statusItem);

    // 3. Register Extension Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('neuron.check', () => {
            const filePath = getActiveFilePath();
            if (filePath) {
                runInTerminal(`"${compilerPath}" check "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('neuron.run', () => {
            const filePath = getActiveFilePath();
            if (filePath) {
                runInTerminal(`"${compilerPath}" run "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('neuron.jit', () => {
            const filePath = getActiveFilePath();
            if (filePath) {
                runInTerminal(`"${compilerPath}" jit "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('neuron.transpile', () => {
            const filePath = getActiveFilePath();
            if (filePath) {
                const pyOut = filePath.replace(/\.nr$/, '.py');
                runInTerminal(`"${compilerPath}" transpile "${filePath}" -o "${pyOut}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('neuron.restartServer', async () => {
            if (client) {
                vscode.window.showInformationMessage('Restarting NEURON Language Server...');
                try {
                    await client.stop();
                    await client.start();
                    vscode.window.showInformationMessage('NEURON Language Server restarted successfully.');
                } catch (e) {
                    vscode.window.showErrorMessage(`Failed to restart NEURON Language Server: ${e.message}`);
                }
            }
        })
    );

    console.log('[NEURON] Extension activated successfully.');
}

function deactivate() {
    if (client) {
        return client.stop();
    }
}

module.exports = { activate, deactivate };
