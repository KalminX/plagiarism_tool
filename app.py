from flask import Flask, request, jsonify
from plagiarism_checker import analyze_text
import asyncio

from flask import Flask, request, jsonify
import traceback

app = Flask(__name__)

@app.route('/check', methods=['POST'])
def check_plagiarism():
    try:
        data = request.get_json()
        text = data.get("text", "")
        print(f"---------------- DATA ---------------\n{text}")
        result = asyncio.run(analyze_text(text))
        return jsonify(result)
    except Exception as e:
        print("❌ Error during /check:")
        traceback.print_exc()  # This prints the full error trace to Heroku logs
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
