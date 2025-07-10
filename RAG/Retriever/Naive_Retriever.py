from langchain_chroma import Chroma
from utils import embeddings_model, CHROMA_PERSIST_DIRECTORY, DB_NAME




def get_Naive_Retriever(top_k):
    """
    Fonction pour obtenir un retriever naïf à partir de la base de données Chroma.
    Args:
        top_k (int): Nombre de documents à récupérer.
    """
    # Recharger la base Chroma
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY, 
        collection_name=DB_NAME,
        embedding_function=embeddings_model
    )


    retriever = vectorstore.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": top_k}     
    )
    return retriever