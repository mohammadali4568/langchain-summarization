from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")


# Load environment variables
load_dotenv()

# Function to create memory based on type
def create_memory(memory_type, llm):
    if memory_type == "Buffer":
        return ConversationBufferMemory(k=3)
    elif memory_type == "Summary":
        return ConversationSummaryMemory(llm=llm)
    else:
        raise ValueError("Invalid memory type. Use 'buffer' or 'summary'.")

# Load LLM configuration
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

# Text samples
machine_learning_text = """
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
These systems improve their performance over time without being explicitly programmed.
Applications of machine learning include recommendation systems, fraud detection, image recognition, and natural language processing.
It uses algorithms like decision trees, support vector machines, and clustering techniques.
Machine learning models require large volumes of structured data for training and evaluation.
Supervised, unsupervised, and reinforcement learning are the main categories of machine learning,
each with distinct use cases. The field is evolving rapidly, with increasing integration in business and technology.
"""

deep_learning_text = """
Deep learning is a specialized subset of machine learning based on artificial neural networks.
It mimics the way the human brain processes information, allowing machines to recognize patterns and make decisions.
Deep learning excels at handling large, unstructured data such as images, audio, and text.
Common applications include speech recognition, image classification, and autonomous driving.
Unlike traditional machine learning, deep learning models automatically extract relevant features from raw data.
Popular architectures include convolutional neural networks (CNNs) and recurrent neural networks (RNNs).
The success of deep learning relies heavily on powerful computing resources and massive datasets for effective training.
"""

# Function to test memory behavior with two summaries
def SummaryByMemoryType(memory_type):
    memory = create_memory(memory_type, llm)
    chain = ConversationChain(llm=llm, memory=memory)

    print("\n\nMemory Type:", memory_type)

    # Summarize Machine Learning text
    prompt1 = f"Please summarize the following text in 3 sentences:\n{machine_learning_text}"
    response1 = chain.invoke(prompt1)['response']
    print("\nMachine learning Summary:", response1)

    # Summarize Deep Learing text
    prompt2 = f"Now, please summarize the following related text in 3 sentences. Consider the prior summary:\n{deep_learning_text}"
    response2 = chain.invoke(prompt2)['response']
    print("\nDeep learning Summary:", response2)

# Run tests for both memory types
SummaryByMemoryType("Buffer")
SummaryByMemoryType("Summary")
