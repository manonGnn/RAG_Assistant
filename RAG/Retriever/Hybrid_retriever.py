from langchain_community.vectorstores import Chroma
from utils import embeddings_model, CHROMA_PERSIST_DIRECTORY, DB_NAME
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.ensemble import ReciprocalRankFusion


def get_hybrid_retriever():
    """
    Returns the hybrid retriever combining BM25 and Chroma retrievers.
    """
    #RETRIEVER 1
    # Recharger la base Chroma
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY, 
        collection_name=DB_NAME,
        embedding_function=embeddings_model
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": 3}     
    )


    #RETRIEVER 2
    texts= vectorstore.get()['documents']
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 3



    # HYBRID RETRIEVER
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[0.5, 0.5]
    )




    return hybrid_retriever
