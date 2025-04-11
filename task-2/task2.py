import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



# env load
load_dotenv()

api_version = os.getenv("API_VERSION")
deployment_name_gpt = os.getenv("DEPLOYMENT_NAME_GPT")
deployment_name_embedding = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
endpoint_url = os.getenv("ENDPOINT_URL")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")


input_text = """Artificial intelligence (AI) is the science of making machines, particularly computer programs, capable of performing tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and decision-making. It involves developing systems that can mimic human cognitive abilities, such as recognizing speech, making decisions, and identifying patterns. 
Key aspects of AI:
Simulating human intelligence:
AI aims to replicate human cognitive abilities in machines, enabling them to perform tasks that require intelligence. 
Data-driven learning:
AI systems learn and improve their performance through exposure to large amounts of data, identifying patterns and relationships. 
Algorithms and models:
AI relies on algorithms and models, which are sets of rules or instructions that guide the AI's analysis and decision-making. 
Variety of applications:
AI has numerous applications, including natural language processing, machine vision, speech recognition, and robotics. 
Ethical considerations:
The development and use of AI raise ethical considerations, such as bias, fairness, and accountability. 
Examples of AI in action:
Self-driving cars:
AI algorithms enable cars to perceive their surroundings, make decisions, and navigate roads autonomously. 
Chatbots and virtual assistants:
AI-powered chatbots and virtual assistants can understand and respond to human language, providing assistance and customer support. 
Medical diagnosis:
AI systems can assist doctors in diagnosing diseases by analyzing medical images and data, according to Merriam-Webster. 
Financial modeling:
AI algorithms can be used to analyze financial data, predict market trends, and manage risk, according to SAS: Data and AI Solutions. """


# azure openai langChain load
llm = AzureChatOpenAI(
    api_key=azure_openai_api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name_gpt,
    api_version=api_version,
)



# prompt template and summarization for 3 sentence summary
prompt_template_3 = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 3 sentences:\n\n{text}",
)


chain_3 = prompt_template_3 | llm



summary_3 = chain_3.invoke({"text": input_text})
print("\n3-sentence summary:\n", summary_3.content)

# prompt template and summarization for 1 sentence summary

prompt_template_1 = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 1 sentence:\n\n{text}",
)


chain_1 = prompt_template_1 | llm

summary_1 = chain_1.invoke({"text":input_text})
print("\n1-sentence summary:\n", summary_1.content)


# comparing results

print("\n\nComparing Resutls\n\n")
print("3-sentence summary:\n", summary_3.content)
print("\n\n1-sentence summary:\n", summary_1.content)

