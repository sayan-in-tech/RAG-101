from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

while(True):
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    relevant_docs = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    Answer the question based on the context provided below. If the answer is not in the context, say "I don't know".
    Context: {context}
    Question: {query}
    Answer:
    """

    result = llm(prompt)[0]['generated_text']

    answer = result.split("Answer:")[-1].strip()

    print(f"Bot: {answer}")