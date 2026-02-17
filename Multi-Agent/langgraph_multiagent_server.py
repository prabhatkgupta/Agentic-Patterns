from fastmcp import FastMCP
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("multi-agent")
client = OpenAI()
model = "anthropic/claude-sonnet-4-5"

@mcp.tool()
def question_answering_tool(question: str) -> str:
	"""Takes a user generic user question and answer it"""

	print(f"in question_answering_tool, {question}")
	messages = [
		{"role": "system", "content": "You are a general question answer agent, which takes simple user questions and provides answer if you are sure."},
		{"role": "user", "content": question}
	]
	content = ""
	
	try:

		response = client.chat.completions.create(
			messages=messages,
			model=model
		)
		content = response.choices[0].message.content
		print(content)
	except Exception as e:
		print(f"exeption {str(e)}")

	return content



@mcp.tool()
def science_tool(science_question: str) -> str:
	"""Takes only a science based question and answer it"""
	print(f"in science_tool, {science_question}")
	messages = [
		{"role": "system", "content": "You are a Science question answer agent, which takes only science based questions and provides answer if you are sure. Only answer science based question, otherwise say I don't know"},
		{"role": "user", "content": science_question}
	]
	content = ""
	try:
	
		response = client.chat.completions.create(
			messages=messages,
			model=model,
		)
		content = response.choices[0].message.content
		print(content)

	except Exception as e:
		print(f"exeption {str(e)}")

	return content


@mcp.tool()
def translation_tool(text: str, language: str):
	"""Takes a text and desired language, returns the translation of the text into that language"""
	print(f"in translation_tool, text={text}, language={language}")
	messages = [
		{"role": "system", "content": "You are an expert in language translation, you are aware of most of the common languages in this world"},
		{"role": "user", "content": f"You are given a text and a desired language, translate the given text into that particular language. \n\nText: {text}\nDesired Language: {language}"}
	]

	content = ""

	try:

		response = client.chat.completions.create(
			messages=messages,
			model=model
		)
		content = response.choices[0].message.content
		print(content)

	except Exception as e:
		print(f"exeption {str(e)}")

	return content




def main():
	mcp.run(transport="streamable-http", host="0.0.0.0", port=9122)


if __name__ == "__main__":
	main()


