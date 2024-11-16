from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains.conversation.memory import ConversationBufferMemory

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

import torch

"""
Main interface to handle file storage and query processing.
"""
class DataStore:

    # initialise the DataStore instance
    def __init__(self, persist_directory="__chroma_db__"):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = None # will hold Chroma instance
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self._initialise_vector_db()
        self._initialise_llm()
    

    # initialise the vector database i.e. Chroma for now
    def _initialise_vector_db(self):
        self.vector_db = Chroma(embedding_function=self.embedding_model, persist_directory=self.persist_directory)

    # initialise LLM
    def _initialise_llm(self):
        # Initialize a small Hugging Face model pipeline
        #model_name = "meta-llama/Llama-3.2-1B"  # or use another lightweight model such as "facebook/opt-1.3b"
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device="cpu",
            max_new_tokens=300,    # Controls the number of generated tokens
            min_length=100,         # Ensures a minimum length for response
            temperature=0.1,
            do_sample=True          # Enables sampling for more natural output
        )  # Use "cpu" for local inference
        
        # Wrap the Hugging Face pipeline with LangChain's HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        self.prompt_template = PromptTemplate(
            input_variables = ["history", "context", "question"],
            template = (
                "Use the following context and history to answer the question.\n"
                "Do not repeat the history or the context or the question in your answer.\n\n"
                "History: {history}\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
                )
        )
        # Define a simple RunnableSequence
        self.qa_chain = RunnableSequence(   
            self.prompt_template,  # Format the prompt
            self.llm,              # Generate text
            (lambda output: output.split("Answer:")[-1].strip())
        )


    # load the file into the vector db
    def load_and_split_pdf(self, file_path, chunk_size=1000, chunk_overlap=100):
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        chunks = text_splitter.split_documents(pages)
        self.vector_db = Chroma.from_documents(documents=chunks, embedding=self.embedding_model, persist_directory=self.persist_directory)


    # search for a relative context within the vector store
    def search(self, query, top_k=3):
        if not self.vector_db:
            print("==> Vector database not initialised")
            return []
        
        results = self.vector_db.similarity_search(query, k=top_k)
        return results
    

    # pass the question and context from the vector database over to LLM
    def get_answer_from_llm(self, query):

        relevant_chunks = self.search(query)
        unique_chunks = list({chunk.page_content for chunk in relevant_chunks})
        context = " ".join(unique_chunks)
                
        # Generate the response using the Hugging Face LLM
        history = self.memory.load_memory_variables({})["history"]
        response = self.qa_chain.invoke({"history":history, "context":context, "question": query})
        self.memory.save_context({"history": history, "question": query}, {"answer": response})

        return response