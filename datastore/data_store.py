from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from accelerate import disk_offload

import torch


class DataStore:

    """
    Main interface to handle file storage and query processing.
    """
    def __init__(self, persist_directory="__chroma_db__"):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = None # will hold Chroma instance
        self._initialise_vector_db()
        self._initialise_llm()
    

    def _initialise_vector_db(self):
        self.vector_db = Chroma(embedding_function=self.embedding_model, persist_directory=self.persist_directory)

    def _initialise_llm(self):
        # Initialize a small Hugging Face model pipeline
        model_name = "meta-llama/Llama-3.2-1B"  # or use another lightweight model such as "facebook/opt-1.3b"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device="cpu",
            max_new_tokens=150,  # Controls the number of generated tokens
            min_length=50,  # Ensures a minimum length for response
            do_sample=False  # Enables sampling for more natural output
        )  # Use "cpu" for local inference
        
        # Wrap the Hugging Face pipeline with LangChain's HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        self.prompt_template = PromptTemplate(
            input_variables = ["context", "question"],
            template = (
                "Context: {context}\n\nQuestion: {question}\n\n"
                "Provide a concise and clear answer only. Do not include the context or the question in your response:"
                )
        )
        self.qa_chain = self.prompt_template | self.llm


    def load_and_split_pdf(self, file_path, chunk_size=1000, chunk_overlap=100):
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        print(f"==> Loaded {len(pages)} pages from PDF")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        chunks = text_splitter.split_documents(pages)
        print(f"==> Pages split into {len(chunks)} chunks")

        self.vector_db = Chroma.from_documents(documents=chunks, embedding=self.embedding_model, persist_directory=self.persist_directory)
        print("==> Saved chunks to Chroma vector database")


    def search(self, query, top_k=3):
        if not self.vector_db:
            print("==> Vector database not initialised")
            return []
        
        results = self.vector_db.similarity_search(query, k=top_k)
        return results
    

    def get_answer_from_llm(self, query):
        relevant_chunks = self.search(query)
        unique_chunks = list({chunk.page_content for chunk in relevant_chunks})
        context = " ".join(unique_chunks)
                
        # Generate the response using the Hugging Face LLM
        response = self.qa_chain.invoke({"context":context, "question": query})
        return response