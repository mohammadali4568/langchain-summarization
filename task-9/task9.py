from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
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

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name_embedding,
    openai_api_version=api_version,
)

with open("../task-3/ai_intro.txt", "r", encoding="utf-8") as file:
    ai_intro_text = file.read()

# it will split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.create_documents([ai_intro_text])

# vector store
vector_store = FAISS.from_documents(docs, embeddings)

# MultiQueryRetriever setup
multi_retriever = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=llm)

# it will retrieve documents for query
multi_docs = multi_retriever.get_relevant_documents("AI advancements")

# it will combine retrieved text
multi_combined_text = "\n".join([doc.page_content for doc in multi_docs])

# summarization prompt
summary_prompt = PromptTemplate(
    template="Summarize the following text in 3 sentences:\n\n{text}",
    input_variables=["text"]
)

# buidl summarization chain
summarizer = LLMChain(llm=llm, prompt=summary_prompt)

# Summarize multi-query retrieved text
multi_summary = summarizer.run(multi_combined_text)

# single query retrival for comaprison
single_docs = vector_store.similarity_search("AI advancements")
single_combined_text = "\n".join([doc.page_content for doc in single_docs])
single_summary = summarizer.run(single_combined_text)

print("\nMulti-Query Summary\n")
print(multi_summary)

print("\nSingle-Query Summary\n")
print(single_summary)
