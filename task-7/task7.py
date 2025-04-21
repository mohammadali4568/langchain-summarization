import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))
from task2 import Summarize
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-3')))
from task3 import TextRetriever

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

api_version = os.getenv("API_VERSION")
deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
deployment_name_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
endpoint_url = os.getenv("ENDPOINT_URL")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

llm = AzureChatOpenAI(
    api_key=azure_openai_api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name_gpt,
    api_version=api_version
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name_embedding,
    openai_api_version=api_version,
)

query = "AI challenges"

qa_chain = load_qa_chain(llm, chain_type="stuff")


pdf_loader = PyPDFLoader("ai_ethics.pdf")
pdf_docs = pdf_loader.load()

# print(pdf_docs[0].page_content[:1000])  # Print the first 1000 characters of the first page
pdf_chunks = splitter.split_documents(pdf_docs)
pdf_vectorstore = FAISS.from_documents(pdf_chunks, embeddings)
pdf_results = pdf_vectorstore.similarity_search(query)
pdf_summary = qa_chain.run(input_documents=pdf_results, question="Summarize the AI challenges mentioned in this content.")

print("\n--- PDF Summary ---\n", pdf_summary)


web_loader = WebBaseLoader("https://www.sap.com/resources/what-is-ai-ethics")
web_docs = web_loader.load()

# print(web_docs[0].page_content[:1000])  # Print the first 1000 characters of the first page

web_chunks = splitter.split_documents(web_docs)
web_vectorstore = FAISS.from_documents(web_chunks, embeddings)
web_results = web_vectorstore.similarity_search(query)
web_summary = qa_chain.run(input_documents=web_results, question="Summarize the AI challenges mentioned in this content.")

print("\n--- Web Summary ---\n", web_summary)
