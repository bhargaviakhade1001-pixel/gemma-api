from flask import Flask, request, jsonify, send_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

HF_TOKEN = os.getenv("HF_TOKEN")


print("HF_TOKEN loaded:", bool(HF_TOKEN))
print("Loading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch.float32
)

print("Qwen model loaded!")


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "prompt" not in data:
        return jsonify({
            "error": "Please provide a prompt"
        }), 400

    prompt = data["prompt"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

    input_length = inputs["input_ids"].shape[1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    return jsonify({
        "response": response
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
