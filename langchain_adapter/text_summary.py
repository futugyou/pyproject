import requests
# need unstructured[all]
from unstructured.partition.pdf import partition_pdf
from io import BytesIO
from typing import Optional
from pydantic import BaseModel, Field


class SummaryRquest(BaseModel):
    pdf_url: str


def read_pdf(pdf_url: str):
    """Read a PDF file from a URL and return its content."""
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_bytes = BytesIO(response.content)
        return partition_pdf(pdf_bytes)
    else:
        return None


def categorize_elements(raw_pdf_elements):
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


def generate_text_summaries(
    texts, tables, summarize_texts=False, config: LangChainOption = LangChainOption()
):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


if __name__ == "__main__":
    request = SummaryRquest(pdf_url="https://pdfobject.com/pdf/sample.pdf")
    pdf = read_pdf(request.pdf_url)
    if pdf:
        texts, tables = categorize_elements(pdf)

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=4000, chunk_overlap=0
        )
        joined_texts = " ".join(texts)
        texts_4k_token = text_splitter.split_text(joined_texts)
        text_summaries, table_summaries = generate_text_summaries(
            texts_4k_token, tables, summarize_texts=True
        )
        print(text_summaries)
