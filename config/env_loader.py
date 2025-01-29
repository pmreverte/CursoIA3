import os
from dotenv import load_dotenv

load_dotenv(override=True)

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_deployment_name = os.getenv("AZURE_OPENNAI_DEPLOYMENT_NAME")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint_embeding = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDINGS")
bing_search_api_key = os.getenv("BING_SEARCH_API_KEY")
azure_openai_embeding_api_key = os.getenv("AZURE_OPENAI_EMBEDINGS_API_KEY")
azure_openai_embeding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDINGS_DEPLOYMENT_NAME")