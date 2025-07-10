#LLM
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3.2", streaming=True)


# Embeddings model
from langchain_huggingface import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                                         model_kwargs={"trust_remote_code":True},
                                         show_progress=True)

#LOAD VALUES ENVIRONNEMENT
from dotenv import load_dotenv
import os
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
DB_NAME = os.getenv("DB_NAME")