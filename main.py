import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


class RESTAPIHandler(BaseHTTPRequestHandler):

    def send_json(self, status, data):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.send_header(
            "Content-Length",
            str(len(response))
        )
        self.end_headers()

        self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.end_headers()

    def do_GET(self):

        if self.path == "/":
            self.send_json(
                200,
                {
                    "message": "AI REST API is running!"
                }
            )
        else:
            self.send_json(
                404,
                {
                    "error": "Endpoint not found"
                }
            )

    def do_POST(self):

        if self.path != "/api/chat":
            self.send_json(
                404,
                {
                    "error": "Endpoint not found"
                }
            )
            return

        try:
            length = int(
                self.headers.get("Content-Length", 0)
            )

            body = self.rfile.read(length)

            data = json.loads(body)

            prompt = data.get("prompt")

            if not prompt:
                self.send_json(
                    422,
                    {
                        "error": "prompt is required"
                    }
                )
                return

            result = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            self.send_json(
                200,
                {
                    "response": result.text
                }
            )

        except Exception as e:

            print("ERROR:", str(e))

            self.send_json(
                500,
                {
                    "error": str(e)
                }
            )


port = int(
    os.environ.get("PORT", 8000)
)

server = HTTPServer(
    ("0.0.0.0", port),
    RESTAPIHandler
)

print(f"REST API running on port {port}")

server.serve_forever()
