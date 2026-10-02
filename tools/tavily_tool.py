from tavily import TavilyClient
import os 
from dotenv import load_dotenv

load_dotenv()

tavily_api_key=os.getenv("TAVILY_API_KEY")

client=TavilyClient(api_key=tavily_api_key)

# searching the query

def tavily_search(question:str)->list[str]:
    response=client.search(
        query=question,
        max_results=5
        )

    results=[]

    for i, r in enumerate(response["results"], 1):
        title   = r.get("title", "Unknown")
        url     = r.get("url", "")
        snippet = r.get("content", "").strip()
        # Keep only the first 300 characters to avoid wall-of-text
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)