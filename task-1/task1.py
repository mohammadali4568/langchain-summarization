import os
from dotenv import load_dotenv

load_dotenv()

api_version = os.getenv("API_VERSION")
deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
deployment_name_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
endpoint_url = os.getenv("ENDPOINT_URL")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

print("API Version:", api_version)
print("Deployment Name:", deployment_name_gpt)
print("Deployment Name for Embedding:", deployment_name_embedding)
print("Endpoint URL:", endpoint_url)
print("Azure OpenAI Key:", "Loaded" if azure_openai_api_key else "Not Loaded")
