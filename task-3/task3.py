import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task-2')))

from task2 import Summarize
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


class TextRetriever:
    def __init__(self, file_path):
        load_dotenv()

        # Load environment variables
        self.api_version = os.getenv("API_VERSION")
        self.deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
        self.deployment_name_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
        self.endpoint_url = os.getenv("ENDPOINT_URL")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        # Load and split text
        self.file_path = file_path
        self.loader = TextLoader(self.file_path, autodetect_encoding=True)
        self.text = self.loader.load()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
            separators="."
        )
        self.chunks = self.text_splitter.create_documents([self.text[0].page_content])

        # Embedding & vector store setup
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.endpoint_url,
            azure_deployment=self.deployment_name_embedding,
            openai_api_version=self.api_version,
        )
        self.vector_store = InMemoryVectorStore.from_documents(self.chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever()

        # Summarizer
        self.summarizer = Summarize(
            api_key=self.azure_openai_api_key,
            endpoint_url=self.endpoint_url,
            deployment_name=self.deployment_name_gpt,
            api_version=self.api_version
        )

    def retrieve_text(self, query):
        retrieved_docs = self.retriever.invoke(query)
        return retrieved_docs[0].page_content

    def summarize_text(self, text):
        return self.summarizer.summarize_1(text)


def main():
    file_path = "./ai_intro.txt"
    text_retriever = TextRetriever(file_path)

    query = "AI milestones"
    retrieved_text = text_retriever.retrieve_text(query)
    print("Retrieved Text:\n", retrieved_text)

    summary = text_retriever.summarize_text(retrieved_text)
    print("\nSummary:\n", summary)


if __name__ == "__main__":
    main()