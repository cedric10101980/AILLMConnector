from flask import Flask, request,  render_template, session, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from logic import run_asyncio_coroutine, process_file, query_logic, chat_logic, process_path, send_html_email_gmail, query_logic_from_prompt

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = Flask(__name__, template_folder='templates/')

app.config['UPLOAD_FOLDER'] = 'uploads/'
ALLOWED_EXTENSIONS = {'csv'}
DEFAULT_MODEL_NAME = 'gpt-3.5-turbo-0125'
DEFAULT_MODEL_NAME=os.getenv('MODEL_NAME', DEFAULT_MODEL_NAME)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    global file_id
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        file.save(filepath)
        # process the file
        process_file(file_id)

        logging.info('File' + filepath + 'uploaded & processed successfully. You can begin querying now. ')

        # read the file contents and update the GPT index
        return  f"File {filepath} uploaded & processed successfully.", 200

@app.route('/send-email', methods=['POST'])
def sendEmail():
    global retrieval_chain
    data = request.json
    emailaddress = data.get('emailAddress')
    campaignName = data.get('campaignName')
    contactId = data.get('name')
    #bestTimeToContact = data.get('bestTimeToContact')

    try:
        result = run_asyncio_coroutine(send_html_email_gmail(emailaddress, campaignName, contactId))
    except Exception as e:
        return jsonify({"error": "Cannot fulfill the request", "details": str(e)}), 500

    return result, 200

@app.route('/querybotwithprompt', methods=['POST'])
def querypromt():
    global retrieval_chain
    data = request.json
    prompt = data.get('prompt')
    model_name = data.get('model')
    campaignName = data.get('campaignName')
    contactId = data.get('contactid')
    if not model_name:
        model_name = DEFAULT_MODEL_NAME

    result = run_asyncio_coroutine(query_logic_from_prompt(prompt, campaignName, contactId, model_name))

    return result, 200

@app.route('/querybot', methods=['POST'])
def query():
    global retrieval_chain
    data = request.json
    query = data.get('query')
    model_name = data.get('model')
    if not model_name:
        model_name = DEFAULT_MODEL_NAME

    result = run_asyncio_coroutine(query_logic(query, model_name))

    return result, 200

@app.route('/chatbot', methods=['POST'])
def chat():
    global retrieval_chain
    data = request.json
    query = data.get('query')
    model_name = data.get('model')
    if not model_name:
        model_name = DEFAULT_MODEL_NAME

    result = run_asyncio_coroutine(chat_logic(query, model_name))

    return result, 200

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

indexSummary = process_path(os.getenv('DOCS_PATH', 'docs'))

if __name__ == '__main__':
    logging.info("Starting app...")
    logging.info('RAG chatbot is running on http://localhost:5100/')
    # Please do not set debug=True in production
    app.run(host="0.0.0.0", port=5100, debug=True)
