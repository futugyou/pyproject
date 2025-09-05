from langchain_community.document_loaders import CSVLoader, DirectoryLoader, TextLoader, PythonLoader

from .option import LangChainOption

def csv():
    loader = CSVLoader(file_path='./langchain_adapter/files/mlb_teams_2012.csv', csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']
    })
    data = loader.load()
    print(data)


def dir():
    # text_loader_kwargs={'autodetect_encoding': True}
    # loader = DirectoryLoader("", glob="**/*.py", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    
    
    loader = DirectoryLoader("./mcp_adapter", glob="**/*.py", loader_cls=PythonLoader)
    docs = loader.load()
    # print(docs)
    doc_sources = [doc.metadata['source']  for doc in docs]
    print(doc_sources)

if __name__ == "__main__":
    config = LangChainOption()
    
    # csv()
    dir()
