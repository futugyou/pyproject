import asyncio

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


async def sample_passthrough(config: LangChainOption):
    runnable = RunnableParallel(
        passed=RunnablePassthrough(),
        extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
        modified=lambda x: x["num"] + 1,
    )

    result = runnable.invoke({"num": 2})
    print(result)


async def retrieval_passthrough(config: LangChainOption):
    embedding = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )
    vectorstore = FAISS.from_texts(["harrison worked at kensho"], embedding=embedding)
    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    result = await retrieval_chain.ainvoke("where did harrison work?")
    print(result)


if __name__ == "__main__":
    asyncio.run(retrieval_passthrough(LangChainOption()))
