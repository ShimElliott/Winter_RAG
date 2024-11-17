import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from langchain_core.documents import Document

from uuid import uuid4

# Indexing the textbook PDF
db_filepath = "../data/psychology.db"

if os.path.exists(db_filepath):
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=db_filepath,
        collection_name="wholeTextbookPsych"
    )
    print("Database exists, loading from filepath.")
else:
    file_path = "../Data/wholeTextbookPsych.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=db_filepath,
        collection_name="wholeTextbookPsych"
    )
    print("Database does not exist, creating new database with filepath:", db_filepath)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create a tool for the textbook retriever
textbook_retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_textbook_content",
    "Search and return information from the psychology textbook."
)
