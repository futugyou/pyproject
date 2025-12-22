import faiss

from datetime import datetime, timedelta
from langchain_core.documents.base import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import (
    MultiQueryRetriever,
    TimeWeightedVectorStoreRetriever,
    SelfQueryRetriever,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.query_constructor.schema import AttributeInfo

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


def self_querying_retriever(question: str, config: LangChainOption):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )

    docs = [
        Document(
            page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
            metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
        ),
        Document(
            page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
            metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
        ),
        Document(
            page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
            metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
        ),
        Document(
            page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
            metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
        ),
        Document(
            page_content="Toys come alive and have a blast doing so",
            metadata={"year": 1995, "genre": "animated"},
        ),
        Document(
            page_content="Three men walk into the Zone, three men walk out of the Zone",
            metadata={
                "year": 1979,
                "rating": 9.9,
                "director": "Andrei Tarkovsky",
                "genre": "science fiction",
                "rating": 9.9,
            },
        ),
    ]
    vectorstore = Chroma.from_documents(docs, embeddings)
    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating", description="A 1-10 rating for the movie", type="float"
        ),
    ]
    document_content_description = "Brief summary of a movie"
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
    )
    return retriever.invoke(question)


def time_weighted_vectorstore(question: str, config: LangChainOption):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )

    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )

    embedding_size = 3072

    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, decay_rate=0.0000000000000000000000001, k=1
    )
    yesterday = datetime.now() - timedelta(days=1)
    retriever.add_documents(
        [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
    )
    retriever.add_documents([Document(page_content="hello foo")])
    return retriever.invoke(question)


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

    # vectordb = vectordb_with_Chroma(
    #     "https://learn.microsoft.com/zh-cn/azure/architecture/ai-ml/", config
    # )
    # # doc = multi_query_retriever("What is ai-concepts?", vectordb, config)
    # doc = multi_query_retriever_with_output("What is ai-concepts?", vectordb, config)
    # print(doc)

    # result = self_querying_retriever("What are some movies about dinosaurs", config)
    # print(result)

    result = time_weighted_vectorstore("hello world", config)
    print(result)
