from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS

from datastore import DataStore

import os

# Load .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# initialise the Datastore
datastore = DataStore()

# directory where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def index_html():
    return render_template('index.html')


# function called when user uploads a file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Call the DataStore method to process the uploaded file
            datastore.load_and_split_pdf(file_path)
            return jsonify({"message": "File uploaded and processed successfully"}), 200
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# function called when user asks a question
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question','').strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        answer = datastore.get_answer_from_llm(question)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error":f"An error occured {str(e)}"}), 500

# start from command prompt
if __name__ == '__main__':
    app.run(port=5000, debug=True)
