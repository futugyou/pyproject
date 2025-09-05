import requests
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    TextLoader,
    PythonLoader,
    UnstructuredURLLoader,
    BSHTMLLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
)

from .option import LangChainOption


def csv():
    loader = CSVLoader(
        file_path="./langchain_adapter/files/mlb_teams_2012.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["MLB Team", "Payroll in millions", "Wins"],
        },
    )
    data = loader.load()
    print(data)


def dir():
    # text_loader_kwargs={'autodetect_encoding': True}
    # loader = DirectoryLoader("", glob="**/*.py", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

    loader = DirectoryLoader("./mcp_adapter", glob="**/*.py", loader_cls=PythonLoader)
    docs = loader.load()
    # print(docs)
    doc_sources = [doc.metadata["source"] for doc in docs]
    print(doc_sources)


def html():
    url = "https://learn.microsoft.com/zh-cn/azure/architecture/ai-ml/#ai-concepts"
    response = requests.get(url)

    response.encoding = "utf-8"
    html_text = response.text

    with open("./langchain_adapter/files/temp.html", "w", encoding="utf-8") as f:
        f.write(html_text)

    loader = UnstructuredHTMLLoader("./langchain_adapter/files/temp.html")
    # loader = BSHTMLLoader("./langchain_adapter/files/temp.html")
    # loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()
    print(docs)
    doc_sources = [doc.metadata["source"] for doc in docs]
    print(doc_sources)


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["sender_name"] = record.get("sender_name")
    metadata["timestamp_ms"] = record.get("timestamp_ms")

    return metadata


def json():
    loader = JSONLoader(
        file_path="./langchain_adapter/files/facebook_chat.json",
        jq_schema=".messages[]",
        content_key="content",
        metadata_func=metadata_func,
    )

    data = loader.load()
    # print(docs)
    doc_sources = [doc.metadata["source"] for doc in docs]
    print(doc_sources)


def mark():
    loader = UnstructuredMarkdownLoader("README.md")

    docs = loader.load()
    print(docs)
    doc_sources = [doc.metadata["source"] for doc in docs]
    print(doc_sources)


if __name__ == "__main__":
    config = LangChainOption()

    # csv()
    # dir()
    # html()
    # json()
    mark()
