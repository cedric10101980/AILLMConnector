import asyncio
import os
import pandas as pd
import json
from bs4 import BeautifulSoup
from markdown import markdown
from pymongo import MongoClient
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
import re
import shutil
from html import escape

if os.environ.get('SSL_CERT_FILE'):
    os.environ['SSL_CERT_FILE'] = os.environ.get('SSL_CERT_FILE')

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
import logging
import sys

import smtplib
import email.message

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# Include all the imports and logic functions from your original script here
#Loading the OpenAI model
DEFAULT_MODEL_NAME = 'gpt-4-0125-preview'
DEFAULT_MODEL_NAME=os.getenv('MODEL_NAME', DEFAULT_MODEL_NAME)

PERSIST_DIR = "./storage"


file_id = None
index = None

def run_asyncio_coroutine(coroutine):
    return loop.run_until_complete(coroutine)


#add a function to process individual files
def process_file(file_id):
    global retrieval_chain
    # Loading the splitting the document #
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)

    docs = SimpleDirectoryReader(filepath).load_data()

    # Load chunked documents into the Qdrant index
    # Load the existing index from the persistent store
    index = VectorStoreIndex.from_documents(docs)

    # Add the new documents to the index
    for doc in docs:
        index.add_document(doc)

    # Save the updated index to the persistent store
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def storeIndexDB(docs):
    try:
        mongodb_uri = os.environ.get('MONGODB_URI')
        if mongodb_uri is None:
            print("Environment variable 'MONGODB_URI' is not set")
            raise ValueError("Environment variable 'MONGODB_URI' is not set")
        
        print("Connecting to MongoDB" + mongodb_uri)
        # Create a client
        client = MongoClient(mongodb_uri)

        # Connect to your database
        db = client['mongodb_container']
        collection = db['vector_embeddings']
        # Delete the vector_embeddings collection
        collection.delete_many({})

        # Insert your document
        # Instantiate the vector store
        atlas_vector_search = MongoDBAtlasVectorSearch(
            client,
            db_name = "mongodb_container",
            collection_name = "vector_embeddings"
        )
        vector_store_context = StorageContext.from_defaults(vector_store=atlas_vector_search)
        vector_store_index = VectorStoreIndex.from_documents(
            docs, storage_context=vector_store_context, show_progress=True)

        return vector_store_index
        # Persist the index
        #vector_store_index.storage_context.persist()

    except Exception as e:
        raise e

def load_index_from_DB():
    mongodb_uri = os.environ.get('MONGODB_URI')
    if mongodb_uri is None:
        print("Environment variable 'MONGODB_URI' is not set")
        raise ValueError("Environment variable 'MONGODB_URI' is not set")
    
    print("Connecting to MongoDB" + mongodb_uri)
    # Create a client
    mongo_client = MongoClient(mongodb_uri)
    # Load the existing index from the persistent store
    vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name="mongodb_container", collection_name="vector_embeddings")
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index

def mask_sensitive_data(doc):
    if isinstance(doc, str):
        # Replace email addresses with first 2 and last 2 characters only
        doc = re.sub(r'\b([A-Za-z0-9._%+-]{2})[A-Za-z0-9._%+-]*([A-Za-z0-9._%+-]{2})@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', r'\1****\2@***.***', doc)
                
        # Replace phone numbers with first 2 and last 2 digits only
        doc = re.sub(r'(\d{2})\d{6}(\d{2})', r'\1****\2', doc)

    return doc

def convert_and_mask(value):
    #return str(value)
    return mask_sensitive_data(str(value))

def campaign_filename_fn(filename):
    try:
        # Remove the file extension
        campaign_name = os.path.basename(filename)
        #filename, _ = os.path.splitext(filename)

        # Split the filename on '__' to get the campaign name and campaign ID
        #campaign_name, campaign_id = filename.split('__')

         # Convert the campaign ID to an integer
        #campaign_id = int(campaign_id)

        # Return a dictionary with the campaign name and campaign ID as metadata
        return {
            "file_name": campaign_name,
            "category": 'Campaign',
            "Interaction Type": 'Outbound',
            "campaignName": campaign_name
        }
    except Exception as e:
        raise e

def contact_filename_fn(filename):
    try:
        # Remove the file extension
        filename = os.path.basename(filename)
        filename, _ = os.path.splitext(filename)

        # Split the filename on '__' to get the contact list name and contact list ID
        contact_list_name, contact_list_id = filename.split('__')

        # Convert the contact list ID to an integer
        contact_list_id = int(contact_list_id)

        # Return a dictionary with the contact list name and contact list ID as metadata
        return {
            "file_name": contact_list_name,
            "category": 'Contact List',
            "contactListName": contact_list_name,
            "contactListId": contact_list_id,  # Convert the contact list ID to an integer
        }
    except Exception as e:
        raise e

def process_path(directory_path):
    global retrieval_chain
    # Add post title and post year as metadata to each chunk associated with a document/transcript   

 # Define the source and destination directories
    src_dir = directory_path
    dst_dir = './maskeddocs'

    # Check if the directory exists
    if os.path.exists(dst_dir):
        # Remove all files and subdirectories in the directory
        shutil.rmtree(dst_dir)

    # Recreate the directory
    os.makedirs(dst_dir, exist_ok=True)

    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(src_dir):
        # Compute the destination directory path
        dst_dirpath = os.path.join(dst_dir, os.path.relpath(dirpath, src_dir))

        # Create the destination directory
        os.makedirs(dst_dirpath, exist_ok=True)

        # For each file, read the content, mask the sensitive data, and save the masked content to the destination directory
        for filename in filenames:
            src_filepath = os.path.join(dirpath, filename)
            dst_filepath = os.path.join(dst_dirpath, filename)

            # Read the CSV file using pandas to get column names
            df = pd.read_csv(src_filepath, encoding='ISO-8859-1')

            # Get column names
            columns = df.columns

            # Read the CSV file again applying the mask function to each value
            df = pd.read_csv(src_filepath, encoding='ISO-8859-1', converters={col: convert_and_mask for col in columns})

            # Save the masked DataFrame to the destination directory
            df.to_csv(dst_filepath, index=False)

    all_docs = []
    # gets filename for the current file being read
    cmpreader = SimpleDirectoryReader(input_dir="./maskeddocs/Call Records", recursive = True) # metadata)
    contactsdocs = cmpreader.load_data()

    all_docs.extend(contactsdocs)

    contactLstReader = SimpleDirectoryReader(input_dir="./maskeddocs/Contact Lists", recursive = True) # metadata)
    ctlstdocs = contactLstReader.load_data()
    all_docs.extend(ctlstdocs)
    
    campaignsReader = SimpleDirectoryReader(input_dir="./maskeddocs/Campaigns", recursive = True) # metadata)
    cmpsdocs = campaignsReader.load_data()
    all_docs.extend(cmpsdocs)

    purchaseReader = SimpleDirectoryReader(input_dir="./maskeddocs/Purchases", recursive = True) # metadata)
    purchadocs = purchaseReader.load_data()
    all_docs.extend(purchadocs)


    print("Number of processed files for learning: ", len(all_docs))

    if os.environ.get('STORE_INDEX') == "true":
        index = VectorStoreIndex.from_documents(all_docs)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        index = storeIndexDB(all_docs)

    return index

def remove_markdown(md_string):
    html = markdown(md_string)
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text()

def format_content(content):
    # Escape HTML to prevent XSS attacks
    content = escape(content)

    # Define patterns for numbers and dates
    number_pattern = re.compile(r'\b\d+\b')  # Simplified number detection
    date_pattern = re.compile(r'\b\d{1,2}/\d{1,2}/\d{2}(\d{2})?\b')  # Detects MM/DD/YYYY and MM/DD/YY  # Simplified date detection, e.g., MM/DD/YYYY

    # Function to wrap matched patterns with span tags
    def wrap_with_span(match, css_class):
        return f'<span class="{css_class}">{match.group(0)}</span>'

    # Apply patterns and wrap content
    content = number_pattern.sub(lambda match: wrap_with_span(match, 'number'), content)
    content = date_pattern.sub(lambda match: wrap_with_span(match, 'date'), content)

    return content

def getPrompt(field, value):
    results = []
    try:
        mongodb_uri = os.environ.get('MONGODB_URI')
        if mongodb_uri is None:
            print("Environment variable 'MONGODB_URI' is not set")
            raise ValueError("Environment variable 'MONGODB_URI' is not set")
        
        print("Connecting to MongoDB" + mongodb_uri)
        # Create a client
        client = MongoClient(mongodb_uri)

        # Connect to your database
        db = client['mongodb_container']

        # Get your collection
        collection = db['promptData']

        print("Fetching Records from DB")
            # Get all documents in the collection
        results = collection.find({field: value})

        results_list = list(results)

        if not results_list:
            return ""
        
        results_str = results_list[0]['prompt'] if results_list else None

        print("Results from MongoDB: ", results_str)

        return str(results_str)

    except Exception as e:
        raise e
        

async def send_html_email_gmail(recipient, campaignName, contactId, model_name=DEFAULT_MODEL_NAME):
    sender = "Outbound - No Reply <" + os.environ.get('EMAIL_SENDER') + ">"

    password = os.environ.get('EMAIL_PASSWORD')
    if not password:
        raise ValueError("Please set the PASSWORD environment variable")
    
    # Your existing query logic here
    if os.environ.get('STORE_INDEX') == "true":
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        index = load_index_from_DB()

    query_engine = index.as_query_engine(llm=OpenAI(temperature=0, model=model_name))

    query = getPrompt("type", "ADMIN_EMAIL")

    print("Prompt received  from MongoDB: ", query)

    query = query.replace("$$CONTACT_ID$$", contactId)
    query = query.replace("$$CAMPAIGN_NAME$$", campaignName)
    #query = query.replace("$$BEST_TIME_TO_CALL$$", bestTimeToContact)

    response = query_engine.query(query)

    html_content = markdown(response.response)
    
    print(f"Sending email to {sender} and content {html_content}")

    msg = email.message.Message()
    msg['Subject'] = "Outstanding Balance!"
    msg['From'] = sender
    msg['To'] = recipient
    password = password
    msg.add_header('Content-Type', 'text/html')
    msg.set_payload(html_content)
    
    # Connect to the SMTP server and send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            # Login Credentials for sending the mail
            server.login("avayaoutbound@gmail.com", password)
            server.sendmail(msg['From'], [msg['To']], msg.as_string().encode('utf-8'))
            return "Email sent successfully"
    except Exception as e:
        raise

async def query_logic_from_prompt(prompt, campaignName, contactId, model_name=DEFAULT_MODEL_NAME):
    # Your existing query logic here
    if os.environ.get('STORE_INDEX') == "true":
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        index = load_index_from_DB()

    query_engine = index.as_query_engine(llm=OpenAI(temperature=0, model=model_name))

    print("Fetching prompt with name: ", prompt)

    query = getPrompt("name", prompt)
    query = query.replace("$$CONTACT_ID$$", contactId)
    query = query.replace("$$CAMPAIGN_NAME$$", campaignName)

    print("Prompt received  from MongoDB: ", query)

    if not query:
        return "No prompt found for the given name. Please check the prompt name and try again."

    response = query_engine.query(query)

    # CSS styles
    styles = """
    <style>
    .string { color: green; }
    .number { color: blue; }
    .date { color: red; }
    </style>
    """

    # Combine styles with HTML content

    #plain_text = markdown(response.response)
    html_content = markdown(format_content(response.response))

    final_html_content = styles + html_content

    print("Response from openai as query:\n", html_content)
    return final_html_content

async def query_logic(query, model_name=DEFAULT_MODEL_NAME):
    # Your existing query logic here
    if os.environ.get('STORE_INDEX') == "true":
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        index = load_index_from_DB()
        
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(llm=OpenAI(temperature=0, model=model_name))

    response = query_engine.query(query)
    response_text = json.dumps(response.response, indent=4)
    plain_text = remove_markdown(response_text)

    plain_text = plain_text.replace("json", "").replace("\\n", "").replace("\\", "")

    print("Response from openai as query:\n", plain_text)
    return plain_text

async def chat_logic(query, model_name=DEFAULT_MODEL_NAME, temperature=0):
    # Your existing query logic here
    if os.environ.get('STORE_INDEX') == "true":
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        index = load_index_from_DB()

    query_engine = index.as_chat_engine(llm=OpenAI(temperature=temperature, model=model_name))

    response = query_engine.query(query)

    print("Reponse from openai as chat " , response.response)

    return response.response


async def chat_openai_with_mongodb_index(query, model_name=DEFAULT_MODEL_NAME):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        
    index = load_index_from_storage(storage_context)
    # Create a query engine from the index
    chat_engine = index.as_chat_engine(llm=OpenAI(temperature=0, model=model_name), chat_mode="context", streaming=True)

    # Query OpenAI
    response = chat_engine.stream_chat(query)

    return response