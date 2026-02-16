import os
from fastmcp import FastMCP
from tavily import TavilyClient
from typing import List, Dict, Optional, Any
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

mcp = FastMCP("orchestrator-worker")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
client = OpenAI()
model = "moonshotai/Kimi-K2-Thinking:novita"


@mcp.tool
def web_search(
	query: str = Field(..., description="The text query to search on web"),
	max_results: int = Field(default=5, description="The limit on the maximum results to fetch from the Web"),
	topic: Optional[str] = Field(None, description="The topic regarding the query, refines the search results"),
	include_domains: Optional[List[str]] = Field(None, description="Only Allowed domains to search on the web"),
	exclude_domains: Optional[List[str]] = Field(None, description="Domains to exclude from the search results"),
	start_date: Optional[str] = Field(None, description="Only fetch results after this start date, format: DD-MM-YYYY"),
	end_date: Optional[str] = Field(None, description="Only fetch results before this end date, format: DD-MM-YYYY"),
	country: Optional[str] = Field(None, description="Country to ground the results into, search results relevant to that particular country")
) -> Any:
	try:
		print("Calling web-search tool with query: ", query)
		results = tavily_client.search(
			query=query,
			max_results=max_results,
			topic=topic,
			include_domains=include_domains,
			exclude_domains=exclude_domains,
			start_date=start_date,
			end_date=end_date,
			country=country
		)
	except Exception as e:
		print(f"Exception occurred in fetching Web-search results with tavily {str(e)}")
		results = []


	print("Result from web-search: ", results)

	return results




@mcp.tool
def llm_kb_answer(query: str = Field(..., description="The query to generate some answers using a generative LLM")):
	print("Calling llm_kb_answer with query: ", query)

	messages = [
		{"role": "system", "content": "You are a brilliant in solving user query based on your knowledge"},
		{"role": "user", "content": query}
	]

	try:
		response = client.chat.completions.create(
			messages=messages,
			model=model,
		)
		result = response.choices[0].message.content

	except Exception as e:
		print(f"Exception occurred: {str(e)}")
		result = f"Exception : {str(e)}"

	print("Result from llm_kb_answer: ", result)

	return result


def main():
	mcp.run(transport="streamable-http", host="0.0.0.0", port=5000)


if __name__ == "__main__":
	main()

