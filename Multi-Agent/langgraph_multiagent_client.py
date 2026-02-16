from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
import asyncio
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph_swarm import create_handoff_tool, create_swarm
from typing import TypedDict, Optional, Annotated, Callable, Any
from operator import add
import json
from openai import OpenAI

load_dotenv()

client = OpenAI()
model = "anthropic/claude-sonnet-4-5"
max_iterations = 15



class MultiAgentState(TypedDict):
	iterations: int
	session: ClientSession
	messages: list[str]
	answer: str
	query: str
	tool_calls: list[Any]
	handoff_message: str
	current_state: str


class Agent(TypedDict):
	name: str
	system_prompt: str 
	core_tools: list[str]
	handoff_tools: list[dict]
	tool_list_json: list[dict]


def convert_to_openai_schema(tool_data: dict) -> dict:
    """Convert LangChain tool data to OpenAI function schema format."""
    
    # Extract the Pydantic schema from args_schema
    args_schema = tool_data.get('args_schema')
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    return {
        "type": "function",
        "function": {
            "name": tool_data['name'],
            "description": tool_data.get('description', ''),
            "parameters": parameters
        }
    }




async def create_agent(name: str, system_prompt: str, core_tools: list[str], handoff_tools: list[dict], session: ClientSession) -> Callable:

	tools_list = await session.list_tools()
	tools = [
		{
			"type": "function",
			"function": {
				"name": tool.name,
				"description": tool.description, 
				"parameters": tool.inputSchema
			}
		} for tool in tools_list.tools if tool.name in core_tools
	]


	data = Agent(
		name=name,
		system_prompt=system_prompt,
		core_tools=core_tools,
		handoff_tools=handoff_tools,
		tool_list_json=tools + handoff_tools
	)

	async def agentic_node(state: MultiAgentState) -> MultiAgentState:
		iterations = state["iterations"]
		session = state["session"]
		messages = state["messages"]
		query = state["query"]
		handoff_message = ""
		current_state = state["current_state"]


		if len(messages) == 0:
			messages = [
				{"role": "system", "content": data["system_prompt"] + "\nOnce you have all the data as required for the query, consolidate a final answer touching all points of the query in a short and concise way"},
				{"role": "user", "content": query}
			]


		print(messages)
		print("-"*80)
		response = client.chat.completions.create(
			model=model,
			messages=messages,
			tools=data["tool_list_json"],
			tool_choice="auto"
		)


		message = response.choices[0].message

		answer = None

		tool_calls = []
		if message.tool_calls:
			for tool in message.tool_calls:

				if tool.function.name in data["core_tools"]:
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
					]})

					output = await session.call_tool(tool.function.name, json.loads(tool.function.arguments))
					messages.append({
						"role": "tool",
						"tool_call_id": tool.id,
						"content": str(output.structuredContent["result"])
					})

				else:
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
					]})
					handoff_message = f"Handing off next part of execution to {tool.function.name}"
					messages.append({
						"role": "tool",
						"tool_call_id": tool.id,
						"content": handoff_message + f"You are not in {tool.function.name} agent, please continue processing your query"
					})
					break



		if len(message.content.strip()):
			messages.append({"role": "assistant", "content": message.content})

		if not tool_calls:
			answer = message.content


		return {
			"messages": messages,
			"iterations": iterations + 1,
			"handoff_message": handoff_message,
			"answer": answer,
			"tool_calls": tool_calls,
		}

	return agentic_node


def extract_tool_name(text: str) -> str:
    """Extract everything after 'Handing off next part of execution to '."""
    prefix = "Handing off next part of execution to "
    if text.startswith(prefix):
        return text[len(prefix):]
    return ""



def conditional_edges(state: MultiAgentState) -> str:
	iterations = state["iterations"]
	handoff_message = state.get("handoff_message", "")
	answer = state.get("answer")
	tool_calls = state.get("tool_calls", [])
	current_state = state["current_state"]

	if "transfer_to_" in handoff_message:
		return handoff_message.split("transfer_to_")[-1]

	if iterations > max_iterations or answer:
		return "END"

	return "continue"




async def main():

	async with streamable_http_client("http://localhost:9122/mcp") as (read, write, stdio):
		async with ClientSession(read, write) as session:
			await session.initialize()

			question_answering_handoff = create_handoff_tool(agent_name="question_answering_agent")
			question_answering_handoff = convert_to_openai_schema(question_answering_handoff.__dict__)
			science_question_handoff = create_handoff_tool(agent_name="science_agent")
			science_question_handoff = convert_to_openai_schema(science_question_handoff.__dict__)
			translation_handoff = create_handoff_tool(agent_name="translation_agent")
			translation_handoff = convert_to_openai_schema(translation_handoff.__dict__)

			question_answering_agent = await create_agent(
				name="question_answering_agent",
				system_prompt="You are a very good QnA agent, who has lot of knowledge about general facts. If user asks general question you can answer that, Even if you know the answer please use the question answering tool. if User asks questions about science route to science agent or if user wants to translate route to translation agent. IMPORTANT: Only call ONE tool at a time - never call multiple tools simultaneously.",
				core_tools=["question_answering_tool"],
				handoff_tools=[science_question_handoff, translation_handoff],
				session=session
			)

			science_agent = await create_agent(
				name="science_agent",
				system_prompt="You are a specialized agent who answer question only Science topics. If user asks relevant science quesiton answer them, otherwise if it is a general question route to question_answering_agent, or if translation is required route to translation_agent. IMPORTANT: Only call ONE tool at a time - never call multiple tools simultaneously.",
				core_tools=["science_tool"],
				handoff_tools=[question_answering_handoff, translation_handoff],
				session=session
			)

			translation_agent = await create_agent(
				name="translation_agent",
				system_prompt="You are very good at language translation. You will be given a text and language, you have to make th translation of the text into that language. If any QnA is asked route to question_answering_agent, if specific science quesiton is asked route to science_agent. IMPORTANT: Only call ONE tool at a time - never call multiple tools simultaneously.",
				core_tools=["translation_tool"],
				handoff_tools=[science_question_handoff, question_answering_handoff],
				session=session
			)

			graph = StateGraph(MultiAgentState)
			graph.add_node("question_answering_agent", question_answering_agent)
			graph.add_node("science_agent", science_agent)
			graph.add_node("translation_agent", translation_agent)

			graph.add_edge(START, "question_answering_agent")
			graph.add_conditional_edges("question_answering_agent", conditional_edges, {
					"END": END,
					"continue": "question_answering_agent",
					"science_agent": "science_agent",
					"translation_agent": "translation_agent"
			})


			graph.add_conditional_edges("science_agent", conditional_edges, {
					"END": END,
					"continue": "science_agent",
					"question_answering_agent": "question_answering_agent",
					"translation_agent": "translation_agent"
			})


			graph.add_conditional_edges("translation_agent", conditional_edges, {
					"END": END,
					"continue": "translation_agent",
					"science_agent": "science_agent",
					"question_answering_agent": "question_answering_agent"
			})

			compiled_graph = graph.compile()


			with open("langgraph_multiagent_client.png", "wb") as f:
				f.write(compiled_graph.get_graph().draw_mermaid_png())



			initial_object = MultiAgentState(**{
				"query": "Hi Bro",
				"session": session,
				"messages": [],
				"iterations": 0,
				"answer": None,
				"tool_calls": [],
				"current_state": "question_answering_agent"
			})

			response = await compiled_graph.ainvoke(initial_object)
			print("response: ", response)
			
			print("answer: ", response["answer"])


if __name__ == "__main__":
	asyncio.run(main())
