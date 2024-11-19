import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from langchain_core.documents import Document

from uuid import uuid4
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_pdf, page_ranges, output_dir):
    reader = PdfReader(input_pdf)

    for idx, (start_page, end_page) in enumerate(page_ranges):
        writer = PdfWriter()

        for page_num in range(start_page-1, end_page):
            writer.add_page(reader.pages[page_num])

        output_pdf = os.path.join(output_dir, f"chapter{idx+1}.pdf")

        with open(output_pdf, "wb") as f:
            writer.write(f)
        print(f"Chapter {idx+1} saved to {output_pdf}")

# Indexing the textbook PDF
db_filepath = "Data/psychology.db"

chapter_retrieval_tools = []

if os.path.exists(db_filepath):
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=db_filepath,
        collection_name="wholeTextbookPsych"
    )
    print("Database exists, loading from filepath.")
else:
    # Split PDF by chapter
    input_pdf = "Data/wholeTextbookPsych.pdf"
    page_ranges = [
        (19,46),(47,82),(83,120),(121,156),
        (157,192),(193,224),(225,258),(259,290),
        (291,332),(333,370),(371,410),(411,458),
        (459,496),(497,548),(549,610),(611,644)
    ]
    output_dir = "Data"

    split_pdf(input_pdf, page_ranges, output_dir)

    docs = []

    numChaps = 16

    for i in range(1,numChaps+1):
        file_path = f"Data/chapter{i}.pdf"
        loader = PyPDFLoader(file_path)
        full_chapter = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        splits = text_splitter.split_documents(full_chapter)

        for split in splits:
            split.metadata["chapter"] = i
            docs.append(split)

        chap_db_filepath = f"Data/chapter{i}.db"

        if os.path.exists(chap_db_filepath):
            chap_vectorstore = Chroma(
                embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
                persist_directory=chap_db_filepath,
                collection_name=f"chapter{i}"
            )
            print("Chapter " + str(i) + " database exists, loading from filepath.")
        else:
            print("Chapter " + str(i) + " database does not exist, creating new database with filepath:", chap_db_filepath)
            
            chap_vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
                persist_directory=f"Data/chapter{i}.db",
                collection_name=f"chapter{i}"
            )

        chap_retriever = chap_vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 5, "fetch_k": 10})

        chap_tool = create_retriever_tool(
            chap_retriever,
            f"retrieve_chapter{i} content",
            f"Search and return information from chapter {i} of the psychology textbook."
        )

        chapter_retrieval_tools.append(chap_tool)

    # Initialize the vectorstore
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=db_filepath,
        collection_name="wholeTextbookPsych"
    )
    print("Database does not exist, creating new database with filepath:", db_filepath)

retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 8, "fetch_k": 40})

# Create a tool for the textbook retriever
textbook_retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_textbook_content",
    "Search and return information from the psychology textbook."
)
