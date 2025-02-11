from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

reasoning_model = os.getenv("REASONING_MODEL_ID")
tool_model = os.getenv("TOOL_MODEL_ID")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_model(model_id):
    using_huggingface = os.getenv("USING_HUGGINGFACE", "False").lower() == "true"
    if using_huggingface:
        return HfApiModel(model_id=model_id, token=huggingface_api_token)
    else:
        return OpenAIServerModel(model_id=model_id, api_base="http://localhost:11434/v1", api_key="ollama")

# Create the reasoner for better RAG
reasoning_model = get_model(reasoning_model)
reasoner = CodeAgent(tools=[], model=reasoning_model, max_steps=2)

# Initialize vectore store and embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

@tool
def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database.
    The result of the search is given to a reasoning LLM to generate a response, so what you'll get back
    from this tool is a short answer to the user's question based on RAG context.

    Args:
        user_query: The user's question to query the vector database with.
    """

    # Search for relevant content from the vector database
    docs = vectordb.similarity_search(user_query, k=3)

    # Combine the content of the documents into a single string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create a prompt for the reasoning LLM
    prompt = f"""
    Based on the following context, answer the user's question. Be concise and specific.
    If there isn't sufficient information, give as your answer a better query to perform RAG with.
    
    Context:
    {context}

    Question: {user_query}

    Answer:"""

    # Get response from reasoning LLM
    response = reasoner.run(prompt, reset=False)
    return response

# Create the primary agent to direct the RAG process
tool_model = get_model(tool_model)
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner], model=tool_model, max_steps=3)

# Example prompt: Compare and contrast the differences between the two models.
def main():
    ui = GradioUI(primary_agent)
    ui.launch()

if __name__ == "__main__":
    main()