import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

from langchain_core.documents import Document

from uuid import uuid4
from PyPDF2 import PdfReader, PdfWriter

# Might make a class later

NUM_CHAPS = 16
# db_filepath = "Data/psychology.db"
pdoc_filepath = "Data/parent_docs.json"

# Customized child splitter that adds the chapter number
class CustomChildSplitter(RecursiveCharacterTextSplitter):
    def split_documents(self, documents):
        child_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)

            chapter_name = doc.metadata["chapter"]
            for chunk in chunks:
                child_docs.append(
                    Document(
                        page_content= f"{chapter_name}\n{chunk}",
                        metadata=doc.metadata
                    )
                )
        return child_docs

# Splits the PDF into chapters, given a list of page ranges    
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

# Load parent documents from a json file
def load_parent_docs(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            print(f"Loading parent documents from {filepath}")
            json_docs = json.load(f)
            return [
                Document(page_content=doc["page_content"], metadata=doc["metadata"])
                for doc in json_docs
            ]
    print(f"No parent documents found at {filepath}")
    return None

# Save parent documents into a json file
def save_parent_docs(parent_docs, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json_docs = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in parent_docs
        ]
        json.dump(json_docs, f, ensure_ascii=False, indent=4)
        print(f"Parent documents saved to {filepath}")

# Gets parent documents
def get_parent_docs():
    # Gets parent documents
    parent_docs = load_parent_docs(pdoc_filepath)

    # Generates parent documents if they don't exist
    if parent_docs is None:
        output_dir = "Data"
        input_pdf = "Data/wholeTextbookPsych.pdf"

        page_ranges = [
            (19,46),(47,82),(83,120),(121,156),
            (157,192),(193,224),(225,258),(259,290),
            (291,332),(333,370),(371,410),(411,458),
            (459,496),(497,548),(549,610),(611,644)
        ]

        print("No parent documents found, generating new ones...")

        split_pdf(input_pdf, page_ranges, output_dir)

        # Creates parent documents with a large and complete context
        parent_docs = []

        for i in range(1,NUM_CHAPS+1):
            file_path = f"Data/chapter{i}.pdf"
            loader = PyPDFLoader(file_path)
            full_chapter = loader.load()
            parent_docs.append(
                Document(
                    page_content=full_chapter[0].page_content,
                    metadata={"chapter": f"Chapter {i}"}
                )
            )

        save_parent_docs(parent_docs, pdoc_filepath)
        print("Parent documents saved")

        for i in range(1,NUM_CHAPS+1):
            file_path = f"Data/chapter{i}.pdf"
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")

    return parent_docs

parent_docs = get_parent_docs()
        
# Create child document splitter
child_splitter = CustomChildSplitter(
    chunk_size=500,
)
    
# Defines the parent splitter
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
)

# Storage layer for the parent documents
store = InMemoryStore()

# Create the vectorstore
child_vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
    collection_name="wholeTextbookPsych"
)

# Initialize the Parent Document Retriever
retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_type="mmr",
    search_kwargs={"k": 3},
)
retriever.add_documents(parent_docs)

print("Parent Document Retriever initialized")

# Create a tool for the textbook retriever
textbook_retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_textbook_content",
    "Search and return information from the psychology textbook."
)
