"""
This script is a Streamlit application that allows users to upload PDF files, process them to generate embeddings, and interact with a custom agent for querying the content.
Modules:
- logging: For logging information.
- streamlit as st: For creating the Streamlit web application.
- tools.explore_pdf: Contains functions to open, read, and process PDF files.
- tools.search_embedding: Contains functions to create contextual texts, calculate embeddings, and update comparison data.
- tools.search_with_azure: Contains functions to interact with Azure services.
- tools.nuevo_agente: Contains functions to create a new agent.
- tiktoken: For tokenizing text.
- json: For handling JSON data.
- numpy as np: For numerical operations.
- tempfile: For creating temporary files.
Constants:
- TEXT_INPUT_BANNER: A string constant for the text input banner.
Streamlit Components:
- st.title: Sets the title of the Streamlit app.
- st.session_state: Manages the session state for storing variables.
- st.file_uploader: Allows users to upload PDF files.
- st.write: Displays text in the Streamlit app.
- st.button: Creates a button in the Streamlit app.
- st.text_input: Creates a text input field in the Streamlit app.
- st.chat_input: Creates a chat input field in the Streamlit app.
- st.chat_message: Displays chat messages in the Streamlit app.
- st.markdown: Displays markdown text in the Streamlit app.
Functions:
- open_and_read_pdf: Opens and reads a PDF file.
- get_pages_and_texts: Extracts pages and texts from a PDF file.
- concatenate_documents: Concatenates documents based on token size.
- create_contextual_texts_per_pdf: Creates contextual texts for each PDF.
- calculate_embeddings: Calculates embeddings for a given text.
- update_comparision_data: Updates comparison data with embeddings.
- nuevo_agente: Creates a new agent.
"""
import logging
import streamlit as st
from tools.explore_pdf import open_and_read_pdf
from tools.explore_pdf import get_pages_and_texts
from tools.explore_pdf import concatenate_documents
from tools.search_embedding import create_contextual_texts_per_pdf, update_comparision_data
from tools.search_embedding import calculate_embeddings
from tools.search_with_azure import custom_agent_worker
from tools.nuevo_agente import nuevo_agente
import tiktoken
import json
import numpy as np
import tempfile

TEXT_INPUT_BANNER = "¿Sobre qué quieres preguntar?"

st.title("Configuración y prueba de un agente")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.session_state.system_prompt = "Eres un asistente que ayuda. La respuesta siempre la devolverás en el idioma en el que te hablen. Debes de ser breve y conciso en tus respuestas."

st.session_state.chat_history = [{"role": "system", "content": st.session_state.system_prompt}]

uploaded_files = st.file_uploader("Selecciona archivos PDF", type=["pdf"], accept_multiple_files=True)

pages_and_texts=[]
filtered_pages_and_texts=[]
# Verificar si se han cargado archivos
if uploaded_files is not None:
    st.write(f"Se han cargado {len(uploaded_files)} archivos:")
    
    # Iterar a través de los archivos seleccionados y mostrar sus nombres
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
        temp_file_path = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())  # Guardar el archivo subido
            temp_file_path = temp_file.name
        pages_and_texts.append(open_and_read_pdf(temp_file_path))
        filtered_pages_and_texts.append(get_pages_and_texts(pages_and_texts[-1]))
    
    overlap_ratio = 0.2

    overlapped_texts_per_pdf = create_contextual_texts_per_pdf(filtered_pages_and_texts, overlap_ratio=overlap_ratio)

    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    input_texts_and_tokens = [{'file_name': text['file_name'],'text': texto, 'token_size': len(tokenizer.encode(texto))} for text in overlapped_texts_per_pdf for texto in text['texts']]

    max_tokens = 1000

    concatenated_input_texts_and_tokens = concatenate_documents(input_texts_and_tokens, max_tokens)

    text_and_embeddings = [{'block_id': block_id, 'text': text['text'], 'embeddings': calculate_embeddings(text['text'])} for block_id, text in enumerate(concatenated_input_texts_and_tokens)]

    oputput_file = "Embeddings.json"

    with open(oputput_file, "w") as file:
        json.dump(text_and_embeddings, file)

    embedings_list = json.load(open(oputput_file,'r'))

    embedings_list = [{
        "text": entry["text"],
        "embeddings": np.array(entry["embeddings"])
    } for entry in embedings_list]

    update_comparision_data(embedings_list)

else:
    st.write("Por favor, selecciona uno o más archivos PDF.")


if st.button('Instanciar el prompt del sistema'):
    # Solicitar al usuario que ingrese un mensaje después de hacer clic en el botón
    mensaje = st.text_input("Introduce el prompt:")

    # Mostrar el mensaje ingresado
    if mensaje:
        st.session_state.system_prompt = mensaje
    else:
        st.session_state.system_prompt ="Eres un asistente que ayuda. La respuesta siempre la devolverás en el idioma en el que te hablen. Debes de ser breve y conciso en tus respuestas."

    st.session_state.chat_history = [{"role": "system", "content": st.session_state.system_prompt}]
    
    st.session_state.agent = nuevo_agente(st.session_state.system_prompt)

    st.session_state.has_run = False

if "has_run" not in st.session_state or not st.session_state.has_run:
    logger = logging.getLogger("streamlit_app")

    logger.info("Creating a new Agent.")

    from llama_index.llms.azure_openai import AzureOpenAI
    
    st.session_state.agent = nuevo_agente(st.session_state.system_prompt)

    st.session_state.has_run = True
    
if "last_user_prompt" not in st.session_state:
    st.session_state.last_user_prompt = None

user_prompt = st.chat_input(TEXT_INPUT_BANNER)

if user_prompt and user_prompt != st.session_state.last_user_prompt:
    st.session_state.last_user_prompt = user_prompt  # Guarda la última pregunta

    with st.chat_message("user"):
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response = st.session_state.agent.query(user_prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.markdown(response)
