import os
import requests
from flask import Flask, request, jsonify
from gtts import gTTS
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file if exists
load_dotenv()

app = Flask(__name__)

# Load API keys from environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

COHERE_API_URL = "https://api.cohere.ai/v1/generate"

# Function to generate explanation using Cohere API
def generate_dynamic_explanation(topic, level):
    try:
        prompt = f"Provide a {level.lower()}-level explanation of {topic} in data science."
        response = requests.post(
            COHERE_API_URL,
            json={
                "model": "command",
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0.7
            },
            headers={
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        text = response.json()["generations"][0]["text"].strip()
        print(f"[INFO] Generated explanation for {topic} at {level} level.")
        return text
    except Exception as e:
        print(f"[ERROR] Cohere API failed: {e}")
        return f"Fallback explanation for {topic}."

# Function to generate quiz questions using Gemini
def generate_quiz_questions(topic):
    try:
        prompt = f"""
        Generate 6 multiple-choice questions (MCQs) about "{topic}".
        Each question must have 4 options (A, B, C, D) and indicate the correct answer.
        Format:
        Question 1: [Question text]
        A) [Option 1]
        B) [Option 2]
        C) [Option 3]
        D) [Option 4]
        Answer: [Correct option letter]
        """
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        print(f"[INFO] Generated quiz for topic {topic}")
        return parse_quiz_questions(response.text)
    except Exception as e:
        print(f"[ERROR] Gemini API failed: {e}")
        return []

# Parse the quiz text into structured questions
def parse_quiz_questions(text):
    questions = []
    lines = text.split("\n")
    current_question = None

    for line in lines:
        if line.startswith("Question"):
            if current_question:
                questions.append(current_question)
            current_question = {"question": line, "options": [], "answer": ""}
        elif line.startswith(("A)", "B)", "C)", "D)")):
            if current_question:
                current_question["options"].append(line)
        elif line.startswith("Answer:"):
            if current_question:
                current_question["answer"] = line.split("Answer:")[1].strip()
                questions.append(current_question)
                current_question = None

    return questions

# Generate TTS audio
def generate_audio(text, filename):
    try:
        tts = gTTS(text=text, lang='en')
        audio_path = f"static/{filename}.mp3"
        tts.save(audio_path)
        print(f"[INFO] Audio saved to {audio_path}")
        return f"/static/{filename}.mp3"
    except Exception as e:
        print(f"[ERROR] Audio generation failed: {e}")
        return None

# API route to generate explanation
@app.route('/gen', methods=['POST'])
def generate_explanation():
    data = request.json
    topic = data.get("topic")
    level = data.get("level")

    if not topic or not level:
        return jsonify({"error": "Topic and level are required"}), 400

    explanation = generate_dynamic_explanation(topic, level)
    audio_filename = f"{topic}_{level}".replace(" ", "_")
    audio_url = generate_audio(explanation, audio_filename)

    return jsonify({"text": explanation, "audio_url": audio_url})

# API route to generate quiz
@app.route('/quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    topic = data.get("topic")
    questions = generate_quiz_questions(topic)
    return jsonify({"questions": questions})

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001)
