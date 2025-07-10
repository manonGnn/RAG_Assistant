from Retriever.Naive_Retriever import get_Naive_Retriever

import json

def Retrieve_documents(query,top_k):
    """
    Retrieve relevant documents based on the user's query using a naive retriever.
    """
    # 1. Retrieve relevant documents using the naive retriever
    retriever = get_Naive_Retriever(top_k)

    retrieved_docs = retriever.invoke(query)
    # Enregistrer les documents dans un fichier JSON
    docs_to_save = [
        {"page_content": doc.page_content, **getattr(doc, "__dict__", {})}
        for doc in retrieved_docs
    ]
    with open("retrieved_docs.json", "w", encoding="utf-8") as f:
        json.dump(docs_to_save, f, ensure_ascii=False, indent=2)

    return retrieved_docs

if __name__ == "__main__":
    query = "What is the main topic of the document?"
    Retrieve_documents(query)