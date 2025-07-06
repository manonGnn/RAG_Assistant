from langchain_community.document_loaders import PyMuPDFLoader # ou PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from utils import embeddings_model,CHROMA_PERSIST_DIRECTORY, PDF_PATH, DB_NAME
import logging
from tqdm import tqdm

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 1. Loading
logging.info("Chargement du PDF...")
loader = PyMuPDFLoader(file_path =PDF_PATH)
documents = loader.load()
logging.info(f"{len(documents)} pages chargées.")

# 2. Chunking
logging.info("Découpage en chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = []

for doc in tqdm(documents, desc="Découpage des documents"):
    texts.extend(text_splitter.split_documents([doc]))

logging.info(f"{len(texts)} chunks générés.")

# 3. Embeddings
logging.info("Chargement du modèle d'embeddings...")
embeddings = embeddings_model # ou ton modèle perso

# 4. Build vector store database
logging.info("Construction de la base vectorielle avec Chroma...")
vectorstore = Chroma.from_documents(texts, embeddings, collection_name=DB_NAME, persist_directory=CHROMA_PERSIST_DIRECTORY)

logging.info("Base vectorielle Chroma sauvegardée avec succès.")
