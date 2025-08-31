from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_transformers import LongContextReorder
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .option import LangChainOption

prompt_template = """
Given these texts:
-----
{context}
-----
Please answer the following question:
{query}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "query"],
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


def reorder_retrieved_results(retriever, query):
    docs = retriever.invoke(query)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs


def llm_invoke(reordered_docs, query, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    chain = create_stuff_documents_chain(llm, prompt)
    return chain.invoke({"context": reordered_docs, "query": query})


if __name__ == "__main__":
    config = LangChainOption()
    query = "What did the president say about Ketanji Jackson Brown"
    retriever = get_retriever(config)
    reordered_docs = reorder_retrieved_results(retriever, query)
    docs = llm_invoke(reordered_docs, query, config)
    print(docs)
