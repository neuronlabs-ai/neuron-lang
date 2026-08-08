import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';

let diagnosticCollection: vscode.DiagnosticCollection;

interface PyCheckDiagnostic {
    line: number;
    col: number;
    severity: string;
    code: string;
    message: string;
    help?: string;
}

export function activate(context: vscode.ExtensionContext) {
    diagnosticCollection = vscode.languages.createDiagnosticCollection('pycheck');
    context.subscriptions.push(diagnosticCollection);

    // Run on file save
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument((doc) => {
            if (doc.languageId === 'python') {
                analyzeDocument(doc);
            }
        })
    );

    // Run on file open
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument((doc) => {
            if (doc.languageId === 'python') {
                analyzeDocument(doc);
            }
        })
    );

    // Run on active editor change
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor && editor.document.languageId === 'python') {
                analyzeDocument(editor.document);
            }
        })
    );

    // Analyze any already-open Python files
    vscode.workspace.textDocuments.forEach((doc) => {
        if (doc.languageId === 'python') {
            analyzeDocument(doc);
        }
    });

    // Register command to manually run analysis
    context.subscriptions.push(
        vscode.commands.registerCommand('pycheck.analyze', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'python') {
                analyzeDocument(editor.document);
            }
        })
    );

    // Status bar
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBar.text = '$(shield) PyCheck';
    statusBar.tooltip = 'PyCheck — NEURON ML Safety Analyzer';
    statusBar.command = 'pycheck.analyze';
    statusBar.show();
    context.subscriptions.push(statusBar);
}

function analyzeDocument(document: vscode.TextDocument) {
    const config = vscode.workspace.getConfiguration('pycheck');
    const enabled = config.get<boolean>('enabled', true);
    if (!enabled) {
        diagnosticCollection.delete(document.uri);
        return;
    }

    const pythonPath = config.get<string>('pythonPath', 'python');
    const showInfo = config.get<boolean>('showInfo', false);
    const filePath = document.uri.fsPath;

    // Find pycheck module — look relative to the extension or use installed version
    const args = ['-m', 'pycheck', filePath, '--json'];
    if (showInfo) {
        args.push('--info');
    }

    const options: cp.ExecOptions = {
        timeout: 10000,
        maxBuffer: 1024 * 1024,
    };

    cp.exec(`${pythonPath} ${args.join(' ')}`, options, (error, stdout, stderr) => {
        const diagnostics: vscode.Diagnostic[] = [];
        const output = String(stdout);

        if (output.trim()) {
            try {
                const results: PyCheckDiagnostic[] = JSON.parse(output.trim());
                
                for (const result of results) {
                    const line = Math.max(0, result.line - 1);
                    const col = Math.max(0, result.col);
                    
                    // Get the full line to create a proper range
                    const docLine = document.lineAt(line);
                    const range = new vscode.Range(
                        new vscode.Position(line, col),
                        new vscode.Position(line, docLine.range.end.character)
                    );

                    let severity: vscode.DiagnosticSeverity;
                    switch (result.severity) {
                        case 'error':
                            severity = vscode.DiagnosticSeverity.Error;
                            break;
                        case 'warning':
                            severity = vscode.DiagnosticSeverity.Warning;
                            break;
                        default:
                            severity = vscode.DiagnosticSeverity.Information;
                            break;
                    }

                    const message = result.help
                        ? `${result.message}\n\nHelp: ${result.help}`
                        : result.message;

                    const diagnostic = new vscode.Diagnostic(range, message, severity);
                    diagnostic.code = result.code;
                    diagnostic.source = 'PyCheck';
                    diagnostics.push(diagnostic);
                }
            } catch (e) {
                // JSON parse failed — pycheck may have printed non-JSON output
            }
        }

        diagnosticCollection.set(document.uri, diagnostics);
    });
}

export function deactivate() {
    diagnosticCollection?.dispose();
}
