import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

load_dotenv()

api_version = os.getenv("API_VERSION")
deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
endpoint_url = os.getenv("ENDPOINT_URL")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

llm = AzureChatOpenAI(
    api_key=azure_openai_api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name_gpt,
    api_version=api_version
)


def get_current_date(_):
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}."

def static_web_search(query):
    return (
        "Recent AI trends include advancements in large language models, "
        "ethical AI, responsible AI governance, and the use of AI in healthcare, "
        "climate science, and creative industries. Notable developments involve "
        "AI regulations, explainability techniques, and AI-human collaboration tools."
    )

tools = [
    Tool(
        name="Get Date",
        func=get_current_date,
        description="Use this tool to get the current date."
    ),
    Tool(
        name="Web Search",
        func=static_web_search,
        description="Use this tool to search for recent AI trends and updates."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    # verbose=True
)


sample_text = (
    "Artificial Intelligence (AI) is a rapidly growing field focused on developing systems "
    "that can perform tasks requiring human intelligence. From self-driving cars to virtual "
    "assistants and medical diagnostics, AI technologies are transforming industries and "
    "society. The rise of machine learning and deep learning has accelerated AI's impact, "
    "while raising concerns about ethics, data privacy, and job displacement. As AI continues "
    "to evolve, its influence on daily life and global systems is expected to expand, creating "
    "both opportunities and challenges for the future."
)

text_result = agent.invoke(f"Summarize this text: {sample_text} and tell me today's date.")
web_result = agent.invoke("Summarize AI trends and search for recent updates.")

print("\nText: ", text_result['input'])
print("\nSummarize text and show date:\n", text_result['output'])

print("\nWeb Text: ", web_result['input'])
print("\nSumarize web text and show date\n", web_result['output'])
