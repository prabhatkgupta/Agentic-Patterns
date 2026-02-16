import os
from fastmcp import FastMCP
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("react")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The query to search on the web",
    )

    max_results: int = Field(
        5,
        description="Maximum number of relevant search results to return",
    )

    topic: Optional[str] = Field(
        None,
        description=(
            "Topic of the query. This parameter is used to produce a more "
            "strict web search by specifying exact topics."
        ),
    )

    days: Optional[int] = Field(
        None,
        description="Filter for number of days; determines the freshness of the data",
    )

    start_date: Optional[str] = Field(
        None,
        description="Fetch all results after this date (YYYY-MM-DD)",
    )

    end_date: Optional[str] = Field(
        None,
        description="Fetch all results before this date (YYYY-MM-DD)",
    )

    include_domains: Optional[List[str]] = Field(
        None,
        description="Web domains to include in search",
    )

    exclude_domains: Optional[List[str]] = Field(
        None,
        description="Web domains to exclude from search",
    )

    country: Optional[str] = Field(
        None,
        description="Specific country to target the search results from",
    )


@mcp.tool()
def web_search(
	query: str = Field(
        ...,
        description="The query to search on the web",
    ),

    max_results: int = Field(
        5,
        description="Maximum number of relevant search results to return",
    ),

    topic: Optional[str] = Field(
        None,
        description=(
            "Topic of the query. This parameter is used to produce a more "
            "strict web search by specifying exact topics."
        ),
    ),

    days: Optional[int] = Field(
        None,
        description="Filter for number of days; determines the freshness of the data",
    ),

    start_date: Optional[str] = Field(
        None,
        description="Fetch all results after this date (YYYY-MM-DD)",
    ),

    end_date: Optional[str] = Field(
        None,
        description="Fetch all results before this date (YYYY-MM-DD)",
    ),

    include_domains: Optional[List[str]] = Field(
        None,
        description="Web domains to include in search",
    ),

    exclude_domains: Optional[List[str]] = Field(
        None,
        description="Web domains to exclude from search",
    ),

    country: Optional[str] = Field(
        None,
        description="Specific country to target the search results from",
    )
) -> Any:
	"""
	This is a web search tool powered by TavilySearch client, which search the web for information based
	on the query. There are many other tunable parameters that the api accepts. These can be enabled as and when required
	query is the required field
	"""
	print(f"web_search tool called: {query}")

	try:
		response = tavily_client.search(
			query=query,
			max_results=max_results,
			topic=topic,
			days=days,
			start_date=start_date,
			end_date=end_date,
			include_domains=include_domains,
			exclude_domains=exclude_domains,
			country=country
		)
		return response

	except Exception as e:
		return {
			"error": f"Failed to search on the web: {str(e)}",
			"query": query,
			"result": []
		}




@mcp.tool()
def add(num1: float, num2: float) -> float:
	"""This function takes 2 floating point number and returns their addition"""
	print("add tool called")
	return num1+num2



@mcp.tool()
def get_todays_date() -> str:
    """Returns today's date as a in YYYY-MM-DD format"""
    print("get_todays_date tool called")
    return str(datetime.now().date())


def main():
	mcp.run(transport="streamable-http", host="0.0.0.0", port=8889)

if __name__ == "__main__":
	main()



