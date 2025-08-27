
from flask import Flask, request, jsonify, redirect
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()

# Load vectorstore from disk
def load_data_to_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_data_to_vectorstore()

# Initialize memory and LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
google_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define prompt template
prompt_template = """You are a helpful AI assistant for a website FAQ chatbot. 
Answer the question clearly and naturally based on the retrieved context.
Do not mention that the answer is from the documents or refer to the source text.

Context:
{context}

Question:
{question}

Answer in a helpful, polite tone:"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Create conversational retrieval chain
chat_chain = ConversationalRetrievalChain.from_llm(
    llm=google_llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)


# Create Flask app
app = Flask(__name__)




# Flask endpoint for chatbot query
@app.route("/query", methods=["POST"])
def query_chatbot():
    data = request.get_json()
    question = data.get("question", "")
    try:
        response = chat_chain({"question": question})
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)})


# Flask endpoint for root redirect
@app.route("/")
def root():
    return redirect("/query")


# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
