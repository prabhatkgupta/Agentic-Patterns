from langgraph.graph import StateGraph, START, END
from operator import add
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, TypedDict, Annotated

load_dotenv()

client = OpenAI()
model = "azure/gpt-4o"


class ReflectionState(TypedDict):
	query: str
	feedback: str
	messages: Annotated[list[dict], add]
	answer: str
	max_iterations: int




def llm_node(state: ReflectionState) -> ReflectionState:
	query = state["query"]
	messages = state["messages"]
	feedback = state["feedback"]
	print("in llm_node")

	if len(messages) == 0:
		prompt = f"Can you help answer this query to the best of your knowledge: {query}"
		new_message = {"role": "user", "content": prompt}

	else:
		prompt = f"""
		The message contains the query and its response, here is a feedback based on which you
		need to tune your answer. 
		Feedback: {feedback}
		"""
		new_message = {"role": "user", "content": prompt}
	
	messages.append(new_message)


	response = client.chat.completions.create(
		model=model,
		messages=messages
	)

	answer = response.choices[0].message.content
	answer_message = {"role": "assistant", "content": answer}

	return {
		"messages": [new_message, answer_message],
		"answer": answer,
	}



def provide_feedback(state: ReflectionState) -> ReflectionState:
	query = state["query"]
	answer = state["answer"]
	max_iterations = state["max_iterations"]
	print("in provide_feedback")
	prompt = f"""
		For this query {query}, the model has generated an answer {answer}. Please provide some feedback on tone.
		Ask it to improve if required. If all look good generate empty "" feedback
	"""


	response = client.chat.completions.create(
		model=model,
		messages=[{"role": "user", "content": prompt}]
	)
	print("-"*80)

	return {
		"feedback": response.choices[0].message.content,
		"max_iterations": max_iterations + 1
	}


def decision_node(state: ReflectionState) -> str:
	print("in decision_node")
	feedback = state["feedback"]
	print("feedback", feedback)

	max_iterations = state["max_iterations"]
	if feedback == "" or max_iterations > 3:
		return "end"
	return "feedback"



def main():
	graph = StateGraph(ReflectionState)
	graph.add_node("Answer Generator", llm_node)
	graph.add_node("Reflector", provide_feedback)
	graph.add_edge(START, "Answer Generator")
	graph.add_edge("Answer Generator", "Reflector")
	graph.add_conditional_edges("Reflector", decision_node, {
        "feedback": "Answer Generator",
        "end": END
    })

	compiled_graph = graph.compile()

	with open("langgraph_reflection.png", "wb") as f:
		f.write(compiled_graph.get_graph().draw_mermaid_png())



	example = ReflectionState({
			"query": "What is the capital of UAE ?",
			"feedback": None,
			"messages": [], 
			"answer": None,
			"max_iterations": 0
	})
	response = compiled_graph.invoke(example)
	print("response: ", response)


if __name__ == "__main__":
	main()

