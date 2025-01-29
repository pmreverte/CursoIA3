from tools.search_embedding import search_for_info, calculate_embeddings
from tools.bing_search import search_for_data_in_bing
from tools.search_embedding import cosine_similarity
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import FunctionTool
"""
Registers a custom search tool with Azure OpenAI and returns an agent.
The custom search tool first searches for information using embeddings and, if no relevant information is found, 
it falls back to searching for data in Bing. The relevance of the information is determined by calculating the 
cosine similarity between the query and the response.
Returns:
    AgentRunner: An agent runner instance with the registered custom search tool.
Functions:
    custom_agent_worker(query: str) -> str:
        Executes the search tools in order and stops the search if relevant information is found.
        Args:
            query (str): The search query.
        Returns:
            str: The search result, either from embeddings or Bing.
    search_custom:
        A FunctionTool instance created from the custom_agent_worker function.
    agent_worker:
        A FunctionCallingAgentWorker instance created from the search_custom tool and Azure OpenAI client.
    agent:
        An AgentRunner instance created from the agent_worker.
"""
def custom_agent_worker(query: str):
        """Ejecuta las herramientas en orden y detiene la búsqueda si encuentra información."""
        response = search_for_info(query) 
        # Verifica si la respuesta es válida (no None, no cadena vacía)
        if response and isinstance(response, str) and response.strip():
            # Compara la similitud entre la query y la respuesta encontrada
            similarity_score = cosine_similarity(calculate_embeddings(query), calculate_embeddings(response))

            # Si la similitud es baja (por ejemplo, < 0.4), se considera irrelevante y se devuelve None
            print(similarity_score)
            if similarity_score >= 0.4:
                return response
            else:
                print(f"Respuesta irrelevante, similitud: {similarity_score}. Devolviendo None.")
                return search_for_data_in_bing(query)

        return search_for_data_in_bing(query)
