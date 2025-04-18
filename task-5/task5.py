import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
from task2 import Summarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-3')))
from task3 import TextRetriever

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType


class AgentPipeline:
    def __init__(self, task3_file_path):
        load_dotenv()

        self.task3_file_path = task3_file_path


        self.api_version = os.getenv("API_VERSION")
        self.deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
        self.deployment_name_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
        self.endpoint_url = os.getenv("ENDPOINT_URL")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        self.retriever = TextRetriever(self.task3_file_path)
        self.summarizer = Summarize(
            api_key=self.azure_openai_api_key,
            endpoint_url=self.endpoint_url,
            deployment_name=self.deployment_name_gpt,
            api_version=self.api_version
        )

        self.llm = AzureChatOpenAI(
            api_key=self.azure_openai_api_key,
            azure_endpoint=self.endpoint_url,
            azure_deployment=self.deployment_name_gpt,
            api_version=self.api_version,
        )

        self.tools = [
            Tool(
                name="TextRetriever",
                func=self.retrieve_text_tool,
                description="Use this tool to find text in the document based on a query."
            ),
            Tool(
                name="TextSummarizer",
                func=self.summarize_text_tool,
                description="Use this tool to summarize a given text into 3 sentences."
            ),
            Tool(
                name="WordCounter",
                func=self.count_words_tool,
                description="Use this tool to count the number of words in a given text."
            ),
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            # verbose=True                                               # Uncomment for debugging and reasoning
        )

    def retrieve_text_tool(self, query):
        """Tool function to retrieve text based on a query."""          #docsString used for what function is for
        return self.retriever.retrieve_text(query)
    
    def summarize_text_tool(self, text):
        """Tool function to summarize text into 3 sentences."""
        return self.summarizer.summarize_3(text)

    def count_words_tool(self, text):
        """Tool function to count words in a given text."""
        word_count = len(text.split())
        return f"Word count: {word_count}"

    def run_instruction(self, instruction):
        """Run a natural language instruction through the agent."""         
        return self.agent.invoke(instruction)

if __name__ == "__main__":
    print("Warning: \n")
    task3_file_path = "../task-3/ai_intro.txt"
    pipeline = AgentPipeline(task3_file_path)

    instruction = "Find and summarize text about AI breakthroughs from the document, then count the words in the summary."
    result = pipeline.run_instruction(instruction)

    print("\n\nResult:\n")
    print("Input: ", result['input'])
    print("Output: ", result['output'])