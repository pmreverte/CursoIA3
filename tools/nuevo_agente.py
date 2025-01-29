from config.env_loader import (
    azure_openai_endpoint,
    azure_openai_deployment_name,
    azure_openai_api_key
)
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker
from tools.search_with_azure import custom_agent_worker
from llama_index.core.agent import AgentRunner

def nuevo_agente(system_prompt):
    azure_openai_client = AzureOpenAI(
                                engine = azure_openai_deployment_name,
                                azure_endpoint=azure_openai_endpoint,
                                api_key=azure_openai_api_key,
                                api_version="2024-05-01-preview")

    # Registrar el agente
    search_custom = FunctionTool.from_defaults(
        fn=custom_agent_worker, 
        description="Busca información en embeddings y, si no encuentra, en Bing."
    )

    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=[search_custom],  # Solo se registra la herramienta que coordina ambas búsquedas
        llm=azure_openai_client,
        verbose=True,
        system_prompt=system_prompt
    )

    return AgentRunner(agent_worker)