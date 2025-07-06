from Retriever.Naive_Retriever import get_Naive_Retriever
from utils import llm
from langchain.prompts import PromptTemplate

def main(query):
    """
    Main function to run the RAG system.
    """
    # 1. Retrieve relevant documents using the naive retriever
    retriever = get_Naive_Retriever()

    retrieved_docs = retriever.invoke(query)

    # 2. Generate a response using the LLM and the retrieved documents

    prompt_template = """Vous êtes un assistant juridique expert dans le Code civil français.

                Utilisez uniquement les informations ci-dessous pour répondre à la question posée.

                Documents :
                {documents}

                Question :
                {question}

                Répondez de manière claire et concise, en citant les articles pertinents si possible.
                """
    

    prompt_template_final = PromptTemplate(
    input_variables=["documents", "question"],
    template=prompt_template)

    prompt = prompt_template.format(
    documents="\n\n".join([doc.page_content for doc in retrieved_docs]),
    question=query
)


    response = llm.invoke(prompt)

    # 3. Print the response
    print("Response from LLM:", response)

if __name__ == "__main__":
    query = "What is the main topic of the document?"
    main(query)