import os
import requests
from flask import Flask, request, jsonify, send_file
from gtts import gTTS
from dotenv import load_dotenv
import google.generativeai as genai
from io import BytesIO
import tempfile

# Load .env file if exists
load_dotenv()

app = Flask(__name__)

# Load API keys from environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')  # Use the correct model name

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
            },
        )
        response.raise_for_status()
        return response.json()["generations"][0]["text"].strip()
    except Exception as e:
        print(f"Error with Cohere API: {e}")
        return f"Fallback explanation for {topic}."

# Function to generate quiz using Gemini API
def generate_quiz_questions(topic):
    try:
        prompt = f"""
        Generate 5 multiple-choice questions (MCQs) about "{topic}" in data science.
        For each question:
        1. Provide the question text
        2. Provide 4 options labeled A) to D)
        3. Clearly indicate the correct answer with "Answer: X"
        
        Format each question like this:
        Question: [question text]
        A) [option A]
        B) [option B]
        C) [option C]
        D) [option D]
        Answer: [correct option letter]
        """
        response = model.generate_content(prompt)
        return parse_quiz_questions(response.text)
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return []

# Improved quiz question parser
def parse_quiz_questions(text):
    questions = []
    current_question = {}
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Question:'):
            if current_question:
                questions.append(current_question)
            current_question = {
                'question': line.replace('Question:', '').strip(),
                'options': [],
                'answer': ''
            }
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            current_question['options'].append(line)
        elif line.startswith('Answer:'):
            current_question['answer'] = line.replace('Answer:', '').strip()
            questions.append(current_question)
            current_question = {}
    
    # Add the last question if exists
    if current_question:
        questions.append(current_question)
        
    return questions[:5]  # Return max 5 questions

# Improved audio generation that works in cloud environments
def generate_audio(text):
    try:
        # Create audio in memory
        tts = gTTS(text=text, lang='en')
        audio_io = BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        return audio_io
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

# API Route to generate explanation
@app.route('/gen', methods=['POST'])
def generate_explanation():
    data = request.json
    topic = data.get("topic")
    level = data.get("level")

    if not topic or not level:
        return jsonify({"error": "Topic and level are required"}), 400

    explanation = generate_dynamic_explanation(topic, level)
    audio_io = generate_audio(explanation)

    if audio_io:
        # For Render deployment, we'll return the audio data directly
        return send_file(
            audio_io,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name=f"{topic}_{level}.mp3"
        )
    else:
        return jsonify({"text": explanation, "audio_url": None})

# API route to generate quiz
@app.route('/quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    topic = data.get("topic")
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
        
    questions = generate_quiz_questions(topic)
    return jsonify({"questions": questions})

# Health check endpoint for Render
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001)
