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

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)


def generate_embedding(input_texts: List[str], config: LangChainOption):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )
    return embeddings_model.embed_documents(input_texts)


def retriever(question: str, path: str, config: LangChainOption):
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
    return retriever.invoke(question)


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


def multi_query_retriever_with_output(question: str, vectordb, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    llm_chain = QUERY_PROMPT | llm | output_parser

    # MultiQueryRetriever.from_llm is implemented like this
    retriever_from_llm = MultiQueryRetriever(
        retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
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

    # doc = retriever("what is `Structure`?", "./README.md", config)
    # print(doc)

    vectordb = vectordb_with_Chroma(
        "https://learn.microsoft.com/zh-cn/azure/architecture/ai-ml/", config
    )
    # doc = multi_query_retriever("What is ai-concepts?", vectordb, config)
    doc = multi_query_retriever_with_output("What is ai-concepts?", vectordb, config)
    print(doc)
