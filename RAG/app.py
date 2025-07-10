from urllib import response
import chainlit as cl
from main import Retrieve_documents
from utils import llm
from langchain.prompts import PromptTemplate

@cl.on_chat_start
async def start():
    """
    Fonction appelée au démarrage du chat.
    Elle envoie un message de bienvenue à l'utilisateur.
    """
    await cl.Message(
        content="Bienvenue dans l'assistant juridique RAG ! Posez-moi vos questions sur le Code civil français et je vous aiderai à trouver les réponses.",
        author="RAG Assistant"  # Nom de l'assistant
    ).send()

def call_llm(query, llm, retrieved_docs):
    """
    Fonction pour appeler le modèle LLM avec le prompt donné.
    """
    prompt_template = """Vous êtes un assistant juridique expert dans le Code civil français.

                Utilisez seulement les informations ci-dessous pour répondre à la question posée.

                Documents :
                {documents}

                Question :
                {question}

                Répondez de manière claire et concise, en citant les articles pertinents si possible.
                N'inventez pas d'informations, utilisez uniquement les documents fournis.
                """

    prompt = PromptTemplate(
        input_variables=["documents", "question"],
        template=prompt_template
    ).format(
        documents="\n\n".join([doc.page_content for doc in retrieved_docs]),
        question=query
    )

    # response = llm.invoke(prompt)
    response = llm.invoke(prompt)
    return response


@cl.on_message
async def main(message: cl.Message):
    """
    Fonction principale pour traiter les messages de l'utilisateur.
    """
    query = message.content
    retrieved_docs= Retrieve_documents(query, top_k=5)
    # 2. Generate a response using the LLM and the retrieved documents
    response =call_llm(query, llm, retrieved_docs)
    await cl.Message(content=response).send()
    