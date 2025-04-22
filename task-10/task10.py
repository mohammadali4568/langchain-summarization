import os 
import sys
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
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

file_path = "../task-3/ai_intro.txt"
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()[0].page_content

# print("original text: ",docs)

prompt_template = PromptTemplate(
    input_variables=['text'],
    template="Summarize the following text:\n\n{text}"
)

summarized_text_chain = prompt_template | llm

summarized_text = summarized_text_chain.invoke({"text": docs}).content
# print("Summarized text: ",summarized_text)

qa_prompt_template = PromptTemplate(
    input_variables=['text', 'question'],
    template="Answer the following question based on the text:\n\n{text}\n\nQuestion: {question}\n\nAnswer: "
)

qa_chain = qa_prompt_template | llm

question = "What's the key event mentioned?"
print("Question: ",question)
summarized_answer = qa_chain.invoke({"text": summarized_text, "question": question}).content
print("Answer from Summarized Text: ",summarized_answer)

original_text_answer = qa_chain.invoke({"text": docs, "question": question}).content
print("Answer from Original Text: ",original_text_answer)
