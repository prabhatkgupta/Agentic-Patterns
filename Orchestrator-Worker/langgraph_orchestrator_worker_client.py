from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import OpenAI
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession, StdioServerParameters
import asyncio
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Literal, Callable, Awaitable, operator, Annotated
from enum import StrEnum
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()
model = "moonshotai/Kimi-K2-Thinking:novita"


class AgentType(StrEnum):
	TAVILY = "tavily"
	KNOWLEDGE_BASE = "llm_knowledge_base"


class Worker(BaseModel):
	name: AgentType = Field(..., description="Name of the worker")
	query: str = Field(..., description="Sub-query for the worker to process")


class Workers(BaseModel):
	workers: List[Worker]


class OrchestratorWorkerState(TypedDict):
	query: str
	workers: List[dict]
	active_workers: List[Worker]
	messages: List[str]
	result: Annotated[List[dict], operator.add]
	session: ClientSession
	final_answer: str



class AgentParameters(TypedDict):
	agent_name: str
	agent_type: AgentType
	system_prompt: str
	tool_names: List[str]
	tool_call_json: List[dict]
	model_name: str



async def create_agent(
	agent_name: str, 
	agent_type: AgentType,
	system_prompt: str, 
	tool_names: List[str], 
	model_name: str,
	session: ClientSession
) -> Callable[OrchestratorWorkerState, Awaitable[OrchestratorWorkerState]]:
	
	available_tools = await session.list_tools()
	tools = [
		{
			"type": "function",
			"function": {
				"name": tool.name,
				"description": tool.description,
				"parameters": tool.inputSchema
			}
		} for tool in available_tools.tools if tool.name in tool_names
	]

	agent_parameters = AgentParameters(
		agent_name=agent_name,
		agent_type=agent_type,
		system_prompt=system_prompt,
		tool_names=tool_names,
		model_name=model_name,
		tool_call_json=tools
	)


	async def agent_creation_helper(state: OrchestratorWorkerState) -> OrchestratorWorkerState:
		query = state["query"]
		messages = state.get("messages", [])
		answer = None
		print(f"Calling: {agent_parameters['agent_type']}")


		while True:
			if not messages:
				messages = [
					{"role": "system", "content": agent_parameters["system_prompt"]},
					{"role": "user", "content": query}
				]

			print(messages)
			print("-"*80)

			response = client.chat.completions.create(
				messages=messages,
				model=agent_parameters["model_name"],
				tools=agent_parameters["tool_call_json"],
				tool_choice="auto"
			)

			message = response.choices[0].message

			tool_calls = []
			if message.tool_calls:
				for tool in message.tool_calls:
					if tool.function.name in agent_parameters["tool_names"]:
						tool_calls.append({
							"tool_call_id": tool.id,
							"tool_name": tool.function.name,
							"tool_arguments": json.loads(tool.function.arguments)
						})

						messages.append({"role": "assistant", "tool_calls": [
								{
									"id": tool.id,
									"type": "function",
									"function": {
										"name": tool.function.name,
										"arguments": tool.function.arguments
										}
									}
								]
							}
						)

						output = await session.call_tool(tool.function.name, json.loads(tool.function.arguments))

						messages.append({
							"role": "tool",
							"tool_call_id": tool.id,
							"content": str(output)
						})

			else:
				messages.append({"role": "assistant", "content": message.content})
				answer = message.content
				break

		return {
			"result": [{"source": agent_type, "answer": answer}]
		}

	return agent_creation_helper



def orchestrator_node(state: OrchestratorWorkerState) -> OrchestratorWorkerState:
	query = state["query"]
	workers = state["workers"]
	print("In orchestrator Node")
	print("workers:", workers)

	system_prompt = """You are an intelligent LLM based task orchestrator, which can understand the complex user query
and route it to specialized sub-agent or workers. It can either call single or multiple workers depending on the
requirement of the query. First it breaks down the query specifically to be handled by each worker plus generate
the name of worker which will handle this sub-query.


Here are the list of worker and their description to help breakdown the query and routing logic.

Name |  Description:
"""


	for worker in workers:
		system_prompt += f'\n{worker["name"]} | {worker["description"]}'

	system_prompt += """\n\nBreakdown the query into structured format with the following exact JSON schema:
{
    "workers": [
        {
            "name": "tavily" or "llm_knowledge_base",
            "query": "the specific sub-query for this worker"
        }
    ]
}

IMPORTANT: Use exactly these field names: "name" and "query" for each worker object."""


	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": query}
	]


	response = client.chat.completions.parse(
		model=model,
		messages=messages,
		response_format=Workers
	)

	active_workers = response.choices[0].message.parsed
	
	return {
		"active_workers": active_workers.workers
	}



def router(state: OrchestratorWorkerState) -> List[Send]:
	active_workers = state.get("active_workers", [])
	print("In Router Node")
	print("active_workers: ", active_workers, type(active_workers))
	query = state["query"]
	
	return [
		Send(worker.name, {"query": worker.query}) for worker in active_workers
	]


def router_node(state: OrchestratorWorkerState) -> OrchestratorWorkerState:
	return state


def result_node(state: OrchestratorWorkerState) -> OrchestratorWorkerState:
	return state



def synthesizer_node(state: OrchestratorWorkerState) -> OrchestratorWorkerState:
	result = state.get("result", [])
	query = state.get("query")
	print("In Synthesizer Node")


	system_prompt = """You are a synthesizer agent which is able to understand the answers generated from different sources
and create a final response based from them. You are also given the query for which you have to generate the answer"""

	
	if not result:
		if len(state.get("active_workers", [])) == 0:
			formatted_result = "No workers were called for the given query"
		else:
			formatted_result = "Some issue occurred as no results are generated"

	else:
		formatted_result = "\n\n".join([str(res) for res in result])

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": f"Query: {query}\n\nSub-worker results: {formatted_result}"}
	]

	response = client.chat.completions.create(
		model=model,
		messages=messages
	)

	return {"final_answer": response.choices[0].message.content}



async def main():
	async with streamable_http_client("http://localhost:5000/mcp") as (read, write, stdio):
		async with ClientSession(read, write) as session:
			await session.initialize()

			tavily_agent = await create_agent(
				agent_name="Tavily", 
				agent_type=AgentType.TAVILY,
				system_prompt="You are great at web-search. You have access to web-search tool and generate answers from live content on the internet", 
				tool_names=["web_search"], 
				model_name=model,
				session=session
			)


			knowledge_base_agent = await create_agent(
				agent_name="Knowledge-Base",
				agent_type=AgentType.KNOWLEDGE_BASE,
				system_prompt="You are great at generating answers. You are utilizing LLM's training data as KB and generate answers from that",
				tool_names=["llm_kb_answer"],
				model_name=model,
				session=session
			)


			graph = StateGraph(OrchestratorWorkerState)
			graph.add_node("orchestrator", orchestrator_node)
			graph.add_node("router", router_node)
			graph.add_node("web_search_agent", tavily_agent)
			graph.add_node("database_agent", knowledge_base_agent)
			graph.add_node("result", result_node)
			graph.add_node("synthesizer_node", synthesizer_node)
			graph.add_edge(START, "orchestrator")
			graph.add_edge("orchestrator", "router")
			graph.add_conditional_edges("router", router, ["web_search_agent", "database_agent"])
			graph.add_edge("web_search_agent", "result")
			graph.add_edge("database_agent", "result")
			graph.add_edge("result", "synthesizer_node")
			graph.add_edge("synthesizer_node", END)


			compiled_graph = graph.compile()

			with open("orchestrator-worker.png", "wb") as f:
				f.write(compiled_graph.get_graph().draw_mermaid_png())


			response = await compiled_graph.ainvoke(
				OrchestratorWorkerState(
					query="How is the weather in Abu Dhabi tomorrow ? Write a poem on Bird in 5 lines based on your knowledge",
					workers=[
						{"name": AgentType.TAVILY.value, "description": "You are great at web-search. You have access to web-search tool and generate answers from live content on the internet"},
						{"name": AgentType.KNOWLEDGE_BASE.value, "description": "You are great at generating answers. You are utilizing LLM's training data as KB and generate answers from that. Make use of the kb tool"}
					],
					active_workers=[],
					messages=[],
					result=[],
					session=session,
					final_answer=None
				)
			)

			print(response)

			print(f"Final answer: {response['final_answer']}")


if __name__ == "__main__":
	asyncio.run(main())








