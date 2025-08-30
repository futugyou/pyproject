from langchain.chat_models import init_chat_model
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

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


def vectordb_with_Chroma(path: str, config: LangChainOption):
    embedding = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )

    loader = WebBaseLoader(path)
    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    return Chroma.from_documents(documents=splits, embedding=embedding)


def multi_query_retriever(question: str, vectordb, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )
    return retriever_from_llm.invoke(question)


if __name__ == "__main__":
    config = LangChainOption()
    # result = generate_embedding(
    #     [
    #         "Hi there!",
    #         "Oh, hello!",
    #         "What's your name?",
    #         "My friends call me World",
    #         "Hello World!",
    #     ],
    #     config,
    # )
    # print(result)

    # doc = retriever("./README.md", config)
    # print(doc)

    vectordb = vectordb_with_Chroma(
        "https://learn.microsoft.com/zh-cn/azure/architecture/ai-ml/", config
    )
    doc = multi_query_retriever("What is ai-concepts?", vectordb, config)
    print(doc)
