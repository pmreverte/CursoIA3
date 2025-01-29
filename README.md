📌 Proyecto de Búsqueda Inteligente con Agente Conversacional

Este proyecto permite a los usuarios cargar archivos PDF, extraer su contenido, generar embeddings y realizar consultas utilizando un agente conversacional basado en Azure OpenAI y búsqueda en Bing.

📂 Estructura del Proyecto

main.py: Aplicación principal en Streamlit.

env_loader.py: Carga de variables de entorno.

bing_search.py: Realiza búsquedas en Bing.

explore_pdf.py: Procesa PDFs y extrae su contenido.

search_embedding.py: Genera embeddings y busca similitudes.

search_with_azure.py: Coordina búsquedas con embeddings y Bing.

nuevo_agente.py: Configura y ejecuta el agente.

requirements.txt: Dependencias necesarias.

🚀 Instalación y Ejecución

1️⃣ Requisitos Previos

Python 3.8 o superior

Claves API para Azure OpenAI y Bing Search

2️⃣ Instalación de Dependencias

Ejecuta el siguiente comando en la terminal:

pip install -r requirements.txt

3️⃣ Configuración de Variables de Entorno

Crea un archivo .env con el siguiente contenido y completa con tus credenciales:

AZURE_OPENAI_ENDPOINT=...  # URL del endpoint de Azure OpenAI
AZURE_OPENAI_API_KEY=...  # Clave API de Azure OpenAI
AZURE_OPENAI_DEPLOYMENT_NAME=...  # Nombre del despliegue de OpenAI
AZURE_OPENAI_ENDPOINT_EMBEDINGS=...  # URL del endpoint de embeddings de OpenAI
AZURE_OPENAI_EMBEDINGS_API_KEY=...  # Clave API de embeddings de OpenAI
AZURE_OPENAI_EMBEDINGS_DEPLOYMENT_NAME=...  # Nombre del despliegue de embeddings de OpenAI
BING_SEARCH_API_KEY=...  # Clave API para realizar búsquedas en Bing

4️⃣ Ejecución de la Aplicación

Ejecuta la aplicación Streamlit:

streamlit run main.py

🔄 Flujo de Interacción

Carga de PDF: El usuario sube archivos PDF mediante la interfaz de Streamlit.

Procesamiento del PDF:

Se extrae el texto de cada página.

Se generan embeddings mediante Azure OpenAI.

Se almacenan en un JSON para futuras búsquedas.

Creación del Agente: Se instancia un agente de OpenAI configurado para interactuar con los embeddings y la búsqueda en Bing.

Consulta del Usuario:

El usuario ingresa una pregunta en el chat de Streamlit.

El agente busca primero en los embeddings.

Si la similitud es baja, realiza una búsqueda en Bing.

Retorna la mejor respuesta encontrada.

📌 Notas Adicionales

El agente usa cosine similarity para determinar la relevancia de la información encontrada.

Si la similitud es inferior a 0.4, se considera irrelevante y se recurre a Bing.
