from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import BaseOutputParser


from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def get_retriever(config: LangChainOption):
    embedding = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )

    documents = TextLoader("./langchain_adapter/state_of_the_union.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return FAISS.from_documents(texts, embedding).as_retriever()


def contextual_compression(question: str, retriever, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return compression_retriever.invoke(question)


if __name__ == "__main__":
    config = LangChainOption()
    retriever = get_retriever(config)
    doc = contextual_compression(
        "What did the president say about Ketanji Jackson Brown", retriever, config
    )
    pretty_print_docs(doc)
