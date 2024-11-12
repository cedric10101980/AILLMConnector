#!/bin/bash

# Ajay - Hachathon
export OPENAI_API_KEY=sk-puq2A********ZYy
export EMAIL_PASSWORD="****u"
export EMAIL_SENDER="outbound@gmail.com"
export MONGODB_URI="mongodb+srv://mongo:***@aiimageapp.lvklnwl.mongodb.net/mongodb_container?retryWrites=true&w=majority&appName=AIImageAPP"
# Commment out if running outside  Network
export DOCS_PATH=./docs
export PROCESS_DOCS=true
export STORE_INDEX=true
#export SSL_CERT_FILE=certs.pem
export MODEL_NAME=gpt-4-0125-preview
export FLASK_APP=app.py

flask run