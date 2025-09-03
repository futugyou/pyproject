import asyncio
from operator import itemgetter

from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.schema import format_document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

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


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
{context}

Question: {question}
""")

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


async def retrieval_with_memory(config: LangChainOption):
    embedding = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )

    vectorstore = FAISS.from_texts(["harrison worked at kensho"], embedding=embedding)

    retriever = vectorstore.as_retriever()
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )

    memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )
    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | model
        | StrOutputParser(),
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | model,
        "docs": itemgetter("docs"),
    }
    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    return final_chain, memory


if __name__ == "__main__":

    async def run():
        final_chain, memory = await retrieval_with_memory(LangChainOption())
        inputs = {"question": "where did harrison work?"}
        result = final_chain.invoke(inputs)
        print(result["answer"].content)

        memory.save_context(inputs, {"answer": result["answer"].content})
        memory.load_memory_variables({})

        inputs = {"question": "but where did he really work?"}
        result = final_chain.invoke(inputs)
        print(result["answer"].content)

    # asyncio.run(retrieval_passthrough(LangChainOption()))
    asyncio.run(run())
