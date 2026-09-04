"""
Launch the NEURON In-Browser IDE Demo.
Starts a local lightweight HTTP server and opens your default browser automatically.
"""
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
DEMOS_DIR = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DEMOS_DIR, **kwargs)

    def end_headers(self):
        # Allow WASM MIME type and cross-origin isolation
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

def main():
    os.chdir(DEMOS_DIR)
    url = f"http://localhost:{PORT}/demo5_browser_engine.html"
    print(f"\n=======================================================")
    print(f"  NEURON BROWSER IDE LAUNCHER")
    print(f"  Serving 1.19 MB WASM engine at: {url}")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")

    webbrowser.open(url)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
