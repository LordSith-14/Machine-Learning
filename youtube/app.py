from flask import Flask, render_template, request
from pipeline import iterative_rag
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        youtube_url = request.form["youtube_url"]
        question = request.form["question"]

        youtube_api_key = os.getenv("YOU_TUBE_API_KEY")
        gemini_api_key = os.getenv("GOOGLE_API_KEY")

        if not youtube_api_key or not gemini_api_key:
            return "API keys not configured."

        answer = iterative_rag(
            video_url=youtube_url,
            question=question,
            youtube_api_key=youtube_api_key,
            gemini_api_key=gemini_api_key
        )

        return render_template("result.html", answer=answer)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
