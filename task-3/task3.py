import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))

from task2 import Summarize
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

file_path = "./ai_intro.txt"

load_dotenv()

api_version = os.getenv("API_VERSION")
deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
deployment_name_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
endpoint_url = os.getenv("ENDPOINT_URL")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

loader = TextLoader(file_path, autodetect_encoding=True)
text = loader.load()

print(text[0].page_content)
