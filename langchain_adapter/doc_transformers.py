from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
)


from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


def character_text_splitter():
    with open("./langchain_adapter/state_of_the_union.txt") as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.create_documents([state_of_the_union])
    print(texts[0])


def python():
    PYTHON_CODE = """
    def hello_world():
        print("Hello, World!")

    # Call the function
    hello_world()
    """
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=50, chunk_overlap=0
    )
    python_docs = python_splitter.create_documents([PYTHON_CODE])
    print(python_docs)


def markdown():
    markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)
    for split in md_header_splits:
        print(split)


if __name__ == "__main__":
    config = LangChainOption()
    # character_text_splitter()
    # python()
    markdown()
