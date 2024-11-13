from flask import Flask, request
from datastore import DataStore
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

app = Flask(__name__)

datastore = DataStore()

app.config['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
app.config['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
app.config['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
app.config['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')


if __name__ == "__main__":
    # app.run(port=5000)
    datastore.load_and_split_pdf('/Users/deepak/Downloads/Simplify VAT requirements - Product Requirements Document.pdf')
    response = datastore.get_answer_from_llm("what is the expected impact of removing VAT requirements completely?")
    print("====== Response from LLM=======")
    print("===============================")
    print(response)
