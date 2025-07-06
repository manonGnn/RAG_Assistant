from langchain_chroma import Chroma
from utils import embeddings_model, CHROMA_PERSIST_DIRECTORY, DB_NAME




def get_Naive_Retriever():
    # Recharger la base Chroma
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY, 
        collection_name=DB_NAME,
        embedding_function=embeddings_model
    )


    retriever = vectorstore.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": 2}     
    )
    return retriever