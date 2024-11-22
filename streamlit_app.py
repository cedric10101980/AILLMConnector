import streamlit as st
import asyncio
import os
import base64
from pymongo import MongoClient
from streamlit_pills import pills
import numpy as np
import time
from datetime import datetime
from logic import process_path, load_index_from_DB

import os
import httpx

import logging
import sys
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

if os.environ.get('SSL_CERT_FILE'):
    os.environ['SSL_CERT_FILE'] = os.environ.get('SSL_CERT_FILE')

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.mongodb import SimpleMongoReader

PERSIST_DIR = "./storage"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
url = st.secrets["APP_URL"]
encoded_url = base64.b64encode(url.encode()).decode()
app_key = None

if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "gpt-4-0125-preview"

if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.3

if "prompts_list" not in st.session_state:
    st.session_state.prompts_list = []

if "app_key" not in st.session_state:
    st.session_state.app_key = None

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

if 'use_context' not in st.session_state:
    st.session_state.use_context = True

if 'selected_prompt' not in st.session_state:
    st.session_state['selected_prompt'] = ""

# Initialize an empty dictionary

if 'campaign_map' not in st.session_state:
    st.session_state.campaign_map = {}

if 'campaign_name_map' not in st.session_state:
    st.session_state.campaign_name_map = {}

def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "user":
                st.markdown(f'<p style="color:green;">{message["settings"]}</p>', unsafe_allow_html=True)


def add_user_message_to_session(prompt):
    if prompt:
        settings = f"Temperature: {st.session_state['temperature']}, Model: {st.session_state['selected_model']}"
        st.session_state["messages"].append({"role": "user", "content": prompt, "settings": settings})
        with st.chat_message("user"):
            st.markdown(prompt)
            st.markdown(f'<p style="color:green;">{settings}</p>', unsafe_allow_html=True)

def generate_assistant_response(augmented_query):
    primer = """
Your task is to answer user questions based on the information given above each question.It is crucial to cite sources accurately by using the [[number](URL)] notation after the reference. Say "I don't know" if the information is missing and be as detailed as possible. End each sentence with a period. Please begin.
              """
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query},
            ],
            stream=True,
        ):
            if partial_response := response.choices[0].delta.content:
                full_response += partial_response
            message_placeholder.markdown(full_response + "...")
        message_placeholder.markdown(full_response)

        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )
    return full_response

@st.cache_resource(show_spinner=True, ttl=10)
def getPrompts(tick):
    results = []
    try:
        # Create a client
        client = MongoClient(st.secrets["MONGODB_URI"])

        # Connect to your database
        db = client['mongodb_container']

        # Get your collection
        collection = db['promptData']

        if(st.session_state.show_sidebar == 'True'):
            results = collection.find().sort([("type", -1)])
        else:
            # Get all documents in the collection
            results = collection.find({"type": "AGENT"})

    except Exception as e:
        st.error(f"Error occurred while fetching promtps from database: {e}")
        
    return results

@st.cache_resource(show_spinner=True, ttl=5)
def getCallRecords( id ):

    # load objects from mongo and convert them into LlamaIndex Document objects
    # llamaindex has a special class that does this for you
    # it pulls every object in a given collection
    try:
        print(f"Fetching call records for id: {id}")
        #query_dict = { "campaignId" : "{id}"}
        reader = SimpleMongoReader(uri=st.secrets["MONGODB_URI"])
        documents = reader.load_data('mongodb_container',
            'campaign_call_records', # this is the collection where the objects you loaded in 1_import got stored
            field_names=["contactId", "phoneNumber", "timestamp", "callDuration", "completionCode"], # these is a list of the top-level fields in your objects that will be indexed
                                    # make sure your objects have a field called "full_text" or that you change this value
            query_dict={}, # this is a mongo query dict that will filter your data if you don't want to index everything,
            metadata_names=["campaignId", "campaignName"], # these are the top-level fields in your objects that will be stored as metadata
        )
        print(f"call records documents found: {len(documents)}")
    except Exception as e:
        st.error(f"Error occurred while fetching promtps from database: {e}")
        
    return documents



@st.cache_resource(show_spinner=False)
def index_data():
    global retrieval_chain
    #embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    #Settings.embed_model = embed_model
    #Settings.chunk_size = 1024

    process_path("./docs")
   

def createCampaignIdMaps():
    # Read the CSV data into a DataFrame
    df = pd.read_csv('docs/Campaigns/All Campaigns.csv')

    # Convert the DataFrame to a dictionary
    campaign_name_map  = df.set_index('CampaignId')['CampaignName'].to_dict()
    campaign_map = df.set_index('CampaignName')['CampaignId'].to_dict()

    st.session_state.campaign_name_map = campaign_name_map
    st.session_state.campaign_map = campaign_map

def createSideBar():
    with st.sidebar:
        models = {
            "gpt-4o-mini": "🌱",
            "gpt-4-0125-preview": "🤖",
            "gpt-4-turbo-preview": "🚀",
            "gpt-4-1106-preview": "🧠",
            "gpt-3.5-turbo-0125": "🔥",
            "gpt-3.5-turbo": "💡"
        }
            # Create an array of model names
        model_names = [name for name in models.keys()]
        icons = [icon for icon in models.values()]


        st.session_state['selected_model'] = pills(
            "Select the Open AI Model",
            model_names, icons
        )

        if not st.session_state.app_key:
            st.session_state.app_key = st.text_input('Enter your application key:', get_app_key(time.time()).get('api_key'), type="password")

        st.session_state.use_context = st.checkbox('Use Context', True)

        if st.session_state.use_context:
            st.session_state.phone_number = st.text_input('Contact Number :', get_app_key(time.time()).get('phone_number'))
            current_campaign_name = get_app_key(time.time()).get('campaign_name')

            print(f"Current Campaign Name: {current_campaign_name}")
            print(f"{st.session_state.campaign_name_map}")
            
            # If the current app key is in the campaign_name_map, set the default index for the selectbox
            if current_campaign_name in st.session_state.campaign_name_map.values():
                default_index = list(st.session_state.campaign_name_map.values()).index(current_campaign_name)
            else:
                default_index = 0

            # Create a dropdown (selectbox) for campaign names
            st.session_state.campaign_name = st.selectbox(
                'Select Campaign',
                options=list(st.session_state.campaign_name_map.values()),
                index=default_index  # Set the default selection based on the current app key
            )

            #st.session_state.campaign_name = st.text_input('Campaign Name :', )


        st.session_state['temperature'] = st.sidebar.slider('Creativity(Temparature)', min_value=0.0, max_value=1.0, value=0.3, step=0.01)

        # Markdown separator
        # Label
        st.markdown("## Canned Prompts")    

        for index, result in enumerate(st.session_state.prompts_list):
            # Add a button to each column
            if st.button(result['name']):
                print(f"Button clicked with Prompt: {result['prompt']}")
                # Store the selected prompt in the session state
                st.session_state.selected_prompt = result['prompt']
                # Trigger the query processing
                st.session_state.process_query = True

        # Markdown separator
        st.markdown("---")

        if st.button("Reset Chat"):
                        # Clear all messages
            st.session_state["messages"] = []


def generate_chat_response(query):
    try:
        print(f"Generating index for : {query}")
        
        if os.environ.get('STORE_INDEX') == "true":
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
        else:
            index = load_index_from_DB()

        print(f"Index loaded successfully for query: {st.session_state['campaign_name_map']}")

        # Create a query engine from the index
        chat_engine = index.as_chat_engine(llm=OpenAI(temperature=st.session_state['temperature'], model=st.session_state['selected_model']), chat_mode="context", streaming=True, verbose=True)

        # Query OpenAI
        response = chat_engine.stream_chat(query)

    except httpx.HTTPStatusError as e:
        print(f"Found and caught Exception: {str(e)}")
        # Rethrow the exception
        raise Exception(f"Encountered httpx.HTTPStatusError: {str(e)}")

    return response

def process_query(query, app_key, encoded_url):
    if not query.strip():
        st.error(f"Please provide the search query.")
        return
    elif not st.session_state.app_key or not  st.session_state.app_key.strip():  # Check if the key is empty
        st.error(f"Please provide the application key.")
        return
    elif st.session_state.app_key != encoded_url:  # Replace 'expected_key' with the actual key
        st.error(f"The application key is incorrect.")
        return
    elif st.session_state.use_context:
        if not st.session_state.phone_number:
            st.error(f"Please provide the phone number.")
            return
        elif not st.session_state.campaign_name:
            st.error(f"Please provide the campaign name.")
            return
    try: 
        print(f"Starting processing Query: {query}")    
        if st.session_state.use_context:
            if("$$CONTACT_ID$$" not in query and "$$CAMPAIGN_NAME$$" not in query):
                query = f"For Customer with contactid ContactID-{st.session_state.phone_number} in \"{st.session_state.campaign_name}\", {query}"

            query = query.replace("$$CONTACT_ID$$", str(st.session_state.phone_number))
            query = query.replace("$$CAMPAIGN_NAME$$", st.session_state.campaign_name)


        else:
            query = f"{query}"

        print(f"Processing Complete Query: {query}")
       
        # Run the asynchronous function using an event loop
        add_user_message_to_session(query)
        with st.spinner('Thinking...'):
            response = generate_chat_response(query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for token in response.response_gen:
                    if partial_response := token:
                        full_response += partial_response
                    message_placeholder.markdown(full_response + "...")
                message_placeholder.markdown(full_response)

                st.session_state["messages"].append(
                    {"role": "assistant", "content": full_response}
                )
                return full_response
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        st.session_state.button_clicked = False

@st.cache_resource(show_spinner=False)
def get_app_key(tick):
    params = st.query_params

    # Access a parameter
    api_key = params.get('api_key') or ''
    phone_number = params.get('phone_number') or ''
    campaign_name = params.get('campaign_name') or ''
    show_sidebar = params.get('show_sidebar') or ''

    # Combine into one object
    param_object = {
        'api_key': api_key,
        'phone_number': phone_number,
        'campaign_name': campaign_name,
        'show_sidebar': show_sidebar,
    }

    return param_object

def main():
    
    # Streamlit Page Configuration
    st.set_page_config(

        page_title="Assistant",
        page_icon="imgs/avatar_streamly.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    st.session_state.show_sidebar = get_app_key(time.time()).get('show_sidebar')
    st.session_state.app_key = get_app_key(time.time()).get('api_key')
    st.session_state.phone_number = get_app_key(time.time()).get('phone_number')
    st.session_state.campaign_name = get_app_key(time.time()).get('campaign_name')

    if st.session_state.show_sidebar == "True":
        st.header('OpenAI Chatbot for Querying Customer Information', divider='rainbow')
        #st.subheader('Welcome to the OpenAI Chatbot for querying customer information. You can ask me anything about our customers!')
        st.caption('I am Powered by Open AI so may hallucinate a bit. I am your customer service assistant. Please ask me anything about our customers!')
    else:
        st.caption('I am your customer service assistant powered by OpenAI. Click on any of the prompts below to get started or type a query!')

    st.session_state.prompts_list = list(getPrompts(time.time()))
    
    columns = st.columns(len(getattr(st.session_state, 'prompts_list', [])))

    print(f"Should process docs: {os.environ.get('PROCESS_DOCS')}")
    # Check if the environment variable 'process_docs' is set to 'true'
    if os.environ.get('PROCESS_DOCS') == 'true':
        with st.spinner(text="Loading and indexing the customer history – hang tight! This might take a while."):
            index_data()
    
    createCampaignIdMaps()

    if(st.session_state.show_sidebar == 'True'):
        createSideBar()
    
    display_existing_messages()

    query = st.chat_input("Enter your query here")

    print(f"Query: {query}")


    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    if st.session_state.show_sidebar != 'True':
        for index, result in enumerate(st.session_state.prompts_list):
            # Add a button to each column
            if columns[index % 3].button(result['name']):
                print(f"Button clicked with Prompt: {result['prompt']}")
                # Store the selected prompt in the session state
                st.session_state.selected_prompt = result['prompt']
                # Trigger the query processing
                st.session_state.process_query = True

    if query:
        process_query(query, st.session_state.app_key, encoded_url)

    if 'process_query' in st.session_state and st.session_state.process_query:
        process_query(st.session_state.selected_prompt , st.session_state.app_key, encoded_url)
        # Process the query
        # Reset the process_query flag
        st.session_state.process_query = False


if __name__ == "__main__":
    main()
