import requests
from config.env_loader import (
    bing_search_api_key
)
"""
Searches for data using the Bing Search API and returns the snippet of the first result.
Args:
    query (str): The search query string.
Returns:
    str: The snippet of the first search result if available, otherwise "No results found."
Raises:
    requests.exceptions.HTTPError: If the HTTP request returned an unsuccessful status code.
"""
def search_for_data_in_bing(query):
    "Search the Bing Web Search API and return de first result snippet."
    headers = {"Ocp-Apim-Subscription-Key": bing_search_api_key}
    params = {"q": query,"count": 1}

    response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)  
    response.raise_for_status()
    data = response.json()

    if "webPages" in data and "value" in data["webPages"] and len(data["webPages"]["value"]) > 0:
        first_result = data["webPages"]["value"][0]
        snippet = first_result.get("snippet","no snippet")
        return snippet
    else:
        return "No results found."