import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone
from secret_key import pinecone_api_key, pinecone_env

directory = 'C:\\Users\\HP\\ML Models\\langchain\\embed_chatbot\\data'
# embeddings = OpenAIEmbeddings(model_name="ada")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

# query = embeddings.embed_query("Hello World!")

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

index_name = 'pakichatbot'
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)