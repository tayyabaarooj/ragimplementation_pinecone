from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()


loader = PyPDFLoader('ResumeTayyabaArooj.pdf')
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    text_key="text"
)
vectorstore.add_documents(texts)


custom_prompt_template = """Use the database to answer questions:
{context}

Answer the following question:
{question}

Provide an answer below:
"""
prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])


pipe = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",  
    device=-1  
)
llm = HuggingFacePipeline(pipeline=pipe)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5}),
    chain_type_kwargs={"prompt": prompt},
)


def search_resume(query):
    result = qa.invoke({"query": query})  # or {"input": query} depending on LangChain version
    print("Answer:", result['result'])
    return result['result']


query = "What is the funding amount for MelanoDetectAI in HKD?"
search_resume(query)
