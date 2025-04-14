import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))

from task2 import Summarize
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


file_path = "./ai_intro.txt"

load_dotenv()

api_version = os.getenv("API_VERSION")
deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
deployment_name_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
endpoint_url = os.getenv("ENDPOINT_URL")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

loader = TextLoader(file_path, autodetect_encoding=True)
text = loader.load()
# print(loaded_text[0].page_content)   # ai_intro.txt file content

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    separators="."
)
chunks = text_splitter.create_documents([text[0].page_content])

# print(text[0])
# print("\n\n",text[1])         # splitted text

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name_embedding,
    openai_api_version=api_version,
)

# print(embeddings)     # embeddings object

vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)

# print(vector_store)     # vector store object

retriver = vector_store.as_retriever()

retrieved_docs = retriver.invoke("In the 2010s, breakthroughs in deep learning")

print("Retrieved Text:", retrieved_docs[0].page_content)

summarize = Summarize(api_key=azure_openai_api_key, endpoint_url=endpoint_url, deployment_name=deployment_name_gpt, api_version=api_version)

summary_1 = summarize.summarize_1(retrieved_docs[0].page_content)

print("\n\nSummary:\n", summary_1)






