from langgraph.graph import StateGraph, START, END
from operator import add
from typing import Annotated, List, Dict, TypedDict
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
import asyncio
import json
import nest_asyncio
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.runnables.graph import MermaidDrawMethod

load_dotenv()

client = OpenAI()
model = "anthropic/claude-sonnet-4-5"


class ReactState(TypedDict):
	query: str
	messages: list[dict]
	reasoning: str
	tool_calls: bool
	tool_call_responses: list
	answer: str
	iterations: int
	session: ClientSession


async def reasoning_node(state: ReactState) -> ReactState:
	system_prompt = """
	You are given a query, and previous reasoning chains. You are also having access to a tool that helps you better
	answer the query. If it is a complex query, detemine if you require any tool to gather more information. 
	If yes, which tools to call and call them. Once you are done you can use the content returned by tool and your own 
	observations to formulate a sufficient response for the given query. Also generate a reasoning for calling the tools

	If it is a simple query which doesn't require the tool access then you can choose to answer directly.
	"""

	messages = state["messages"]
	query = state["query"]
	reasoning = state["reasoning"]
	iterations = state["iterations"]
	session = state["session"]
	tool_call_responses = state.get("tool_call_responses", [])
	tool_calls = state.get("tool_calls", [])
	print(f"In reasoning_node {iterations}")

	if len(messages) == 0:
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": query}
		]

	elif len(tool_calls) == len(tool_call_responses):
		for tool, response in zip(tool_calls, tool_call_responses):
			messages.append({
				"role": "assistant",
				"tool_calls": [
					{
						"id": tool["tool_call_id"],
						"type": "function",
						"function": {
							"name": tool["tool_name"],
							"arguments": json.dumps(tool["tool_arguments"])
						}
					}
				]
			})
			messages.append({
				"role": "tool",
				"tool_call_id": tool["tool_call_id"],
				"content": str(response)
			})

	tool_list = await session.list_tools()
	tool_list = [
		{
			"type": "function",
			"function": {
				"name": tool.name,
				"description": tool.description, 
				"parameters": tool.inputSchema
			}
		}
		for tool in tool_list.tools
	]

	print(tool_list)
	from time import sleep
	sleep(5)


	response = client.chat.completions.create(
		messages=messages,
		model=model,
		tools=tool_list,
		tool_choice="auto",
	)

	message = response.choices[0].message

	reasoning = None
	answer = None

	tool_calls = []
	if message.tool_calls:
		for tool in message.tool_calls:
			tool_calls.append({
				"tool_call_id": tool.id,
				"tool_name": tool.function.name,
				"tool_arguments": json.loads(tool.function.arguments)
			})


	if message.content:
		reasoning = message.content
		messages.append({"role": "assistant", "content": message.content})

	if not tool_calls:
		answer = message.content

	return {
		"messages": messages,
		"tool_calls": tool_calls,
		"reasoning": reasoning,
		"tool_call_response": [],
		"answer": answer,
		"iterations": iterations + 1
	}


async def action_node(state: ReactState) -> ReactState:
	print(f"In action_node")
	tool_calls = state.get("tool_calls", [])
	print("action_node: ", tool_calls)
	print()
	session = state["session"]

	tool_call_responses = []


	for tool in tool_calls:
		tool_name = tool["tool_name"]
		tool_arguments = tool["tool_arguments"]
		response = await session.call_tool(tool_name, tool_arguments)
		tool_call_responses.append(response)

	return {
		"tool_call_responses": tool_call_responses
	}


async def decision_node(state: ReactState) -> str:
	iterations = state["iterations"]
	max_iterations = 3
	tool_calls = state.get("tool_calls", [])

	print("-"*80)

	if len(tool_calls) == 0 or iterations > max_iterations:
		return "END"
	return "action"






async def main():

	graph = StateGraph(ReactState)
	graph.add_node("reasoning", reasoning_node)
	graph.add_node("action", action_node)
	graph.add_edge(START, "reasoning")
	graph.add_conditional_edges("reasoning", decision_node, {"END": END, "action": "action"})
	graph.add_edge("action", "reasoning")

	compiled_graph = graph.compile()

	nest_asyncio.apply()
	with open("langgraph_tool_call_client.png", "wb") as f:
		f.write(compiled_graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))


	async with streamable_http_client("http://localhost:8889/mcp") as (read, write, close):
		async with ClientSession(read, write) as session:
			await session.initialize()
			initial_object = ReactState(**{
				"query": "What is the weather in Abu Dhabi Today, can you do a sum of the degrees of today and tomorrow's weather",
				"session": session,
				"messages": [],
				"iterations": 0,
				"reasoning": None,
				"answer": None,
				"tool_calls": [],
				"tool_call_responses": []
			})

			response = await compiled_graph.ainvoke(initial_object)
			print("response: ", response)
			print("answer: ", response["answer"])


if __name__ == "__main__":
	asyncio.run(main())

