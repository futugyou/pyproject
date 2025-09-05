from langchain_community.document_loaders import CSVLoader

from .option import LangChainOption

def csv():
    loader = CSVLoader(file_path='./langchain_adapter/files/mlb_teams_2012.csv', csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']
    })
    data = loader.load()
    print(data)

if __name__ == "__main__":
    config = LangChainOption()
    
    csv()
