from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


def generate_embedding(input_texts: List[str], config: LangChainOption):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )
    return embeddings_model.embed_documents(input_texts)


def retriever(path: str, config: LangChainOption):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )

    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings_model)
    retriever = vectorstore.as_retriever()
    return retriever.invoke("what is `Structure`?")


if __name__ == "__main__":
    # result = generate_embedding(
    #     [
    #         "Hi there!",
    #         "Oh, hello!",
    #         "What's your name?",
    #         "My friends call me World",
    #         "Hello World!",
    #     ],
    #     LangChainOption(),
    # )
    # print(result)

    doc = retriever("./README.md", LangChainOption())
    print(doc)
