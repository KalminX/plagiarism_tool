from flask import Flask, request, jsonify
from plagiarism_checker import analyze_text
import asyncio

app = Flask(__name__)

@app.route("/check", methods=["POST"])
def check_document():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON"}), 400

    original_text = data["text"]
    result = asyncio.run(analyze_text(original_text))
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
