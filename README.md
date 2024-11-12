
# AILLMProcessor
This project is built in Python to connect to OPenAI API's and provides RAG Based solution to any third party application which wants to use the benefits of RAG in their application

## Building the Docker Image

You can build the Docker image for this project by running the `createdocker.sh` script:

```bash
./createdocker.sh
```

This script will:

1. Shut down any existing Docker Compose services.
2. Remove the old Docker image.
3. Build a new Docker image using Docker Compose.
4. Run the new Docker container in detached mode.

## Running the Flask Project Locally

To run the Flask project locally, follow these steps:

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Set the `FLASK_APP` environment variable:

```bash
export OPENAI_API_KEY=sk-****wccZYy
export SSL_CERT_FILE=certs.pem
export FLASK_APP=app.py
export MODEL_NAME=gpt-3.5-turbo-0125 
```

3. Run the Flask development server:

```bash
flask run
```

This will start the Flask development server, and you can access the application at `http://localhost:5000`.
```

Create a Vector Index
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 1536,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
