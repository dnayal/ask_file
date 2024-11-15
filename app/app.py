from flask import Flask, request
from datastore import DataStore
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

app = Flask(__name__)

datastore = DataStore()

if __name__ == "__main__":
    # app.run(port=5000)
    datastore.load_and_split_pdf('/Users/deepak/Downloads/Driving test booking receipt.pdf')
    response = datastore.get_answer_from_llm("what is the driving license number?")
    print("====== Response from LLM=======")
    print("===============================")
    print(response)
