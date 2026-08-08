import http.server
import socketserver
import json
import subprocess
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.expanduser(r"~\.ollama\models\blobs\sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816")
NEURONC_PATH = os.path.abspath(os.path.join(DIRECTORY, "..", "target", "release", "neuronc.exe"))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/api/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            prompt = req.get('prompt', 'Hello')

            print(f"[Web AI] Received prompt: {prompt}")

            # Create temporary script file
            script_content = f'fn main():\n  let reply = generate_reply("{prompt}")\n  print(reply)\n'
            temp_script = os.path.join(DIRECTORY, "temp_prompt.nr")
            with open(temp_script, "w", encoding="utf-8") as f:
                f.write(script_content)

            # Run neuronc
            env = os.environ.copy()
            env["NEURON_GGUF_MODEL"] = MODEL_PATH

            try:
                proc = subprocess.run(
                    [NEURONC_PATH, "run", temp_script],
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=os.path.join(DIRECTORY, ".."),
                    timeout=30
                )
                output = proc.stdout
                # Extract reply
                reply = ""
                for line in output.splitlines():
                    if line.startswith("[NeuronLM]:"):
                        reply = line.replace("[NeuronLM]:", "").strip()
                    elif line.startswith("Q") or not line.startswith("["):
                        if line.strip():
                            reply += " " + line.strip()

                if not reply:
                    reply = "Hello! How can I assist you today?"

                reply = reply.replace("Ġ", " ").replace("▁", " ").strip()

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"reply": reply}).encode('utf-8'))

            except Exception as e:
                print(f"[Web AI] Error: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"reply": f"Error: {e}"}).encode('utf-8'))
        else:
            self.send_error(404)

print(f"NEURON Web AI Server running at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
