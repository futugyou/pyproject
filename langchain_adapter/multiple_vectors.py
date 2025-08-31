from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def create_docs():
    documents = TextLoader("./langchain_adapter/state_of_the_union.txt").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    docs = text_splitter.split_documents(documents)
    return docs


def smaller_chunks(docs, config: LangChainOption):
    embedding = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )

    vectorstore = Chroma(collection_name="full_documents", embedding_function=embedding)

    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    doc_ids = [str(uuid.uuid4()) for _ in docs]

    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    sub_docs = []
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)
    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    return retriever


def summaries(docs, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | llm
        | StrOutputParser()
    )
    summaries = chain.batch(docs, {"max_concurrency": 5})
    return summaries


def retriever_with_summaries(summaries, config: LangChainOption):
    embedding = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )

    vectorstore = Chroma(collection_name="summaries", embedding_function=embedding)

    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    return retriever


if __name__ == "__main__":
    config = LangChainOption()
    docs = create_docs()
    retriever = smaller_chunks(docs, config)
    print(retriever.vectorstore.similarity_search("justice breyer")[0])

    summaries = summaries(docs, config)
    retriever = retriever_with_summaries(summaries, config)
    print(retriever.vectorstore.similarity_search("justice breyer")[0])
