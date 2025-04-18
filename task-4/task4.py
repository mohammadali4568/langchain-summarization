import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))

from task2 import Summarize
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

load_dotenv()

print("Warning: ")
api_version = os.getenv("API_VERSION")
deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
deployment_name_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
endpoint_url = os.getenv("ENDPOINT_URL")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

summarize = Summarize(api_key=azure_openai_api_key, endpoint_url=endpoint_url, deployment_name=deployment_name_gpt, api_version=api_version)

text_summarizer_tool = Tool(
    name="Text Summarizer",
    func=summarize.summarize_3,
    description="Summarizes input text into 3 sentences."
)

agent = initialize_agent(
    tools=[text_summarizer_tool],
    llm=summarize.llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

ai_healthcare_text = """
Artificial Intelligence (AI) is transforming healthcare by improving diagnostics, treatment, and patient care. 
AI-powered systems assist doctors in identifying diseases earlier through advanced imaging analysis and predictive algorithms. 
Machine learning models can analyze large datasets to uncover patterns and recommend personalized treatment plans. 
AI-driven virtual assistants enhance patient engagement, providing instant health advice and appointment scheduling. 
Additionally, AI optimizes hospital operations by predicting patient admissions, managing resources, and reducing costs. 
Robotics powered by AI support surgeons in performing precise, minimally invasive procedures. 
Overall, AI enhances efficiency, accuracy, and accessibility in healthcare, leading to better outcomes and improved patient experiences.
"""

result_1 = agent.invoke(f"Summarize the following text: {ai_healthcare_text}")
print("\n\n", result_1['input'])
print("Summary 1:\n", result_1['output'])

result_2 = agent.invoke("Summarize something interesting")
print("\n\n", result_2['input'])
print("Summary 2:\n", result_2['output'])

