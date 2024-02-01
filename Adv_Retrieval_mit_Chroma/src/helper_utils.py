# Helper utility functions from 
# https://learn.deeplearning.ai/advanced-retrieval-for-ai 

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from tqdm import tqdm
import os
import openai
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)
    

def read_pdf(filename):
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    # Filter the empty strings - exclude empty pages
    return [text for text in pdf_texts if text]

def chunk_texts(texts, chunk_size = 1000, chunk_overlap = 0, 
                tokens_per_chunk = 256, sep = ["\n\n", "\n", ". ", " ", ""]):
    character_splitter = RecursiveCharacterTextSplitter(
    separators = sep,
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(texts))
    
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=chunk_overlap, 
                                                        tokens_per_chunk=tokens_per_chunk)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


def load_chroma(filename, collection_name, embedding_function):
    texts = read_pdf(filename)
    chunks = chunk_texts(texts)

    # chroma_cliet = chromadb.Client()
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    chroma_client.delete_collection(name=collection_name) 
    chroma_collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)

    ids = [str(i) for i in range(len(chunks))]

    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection

def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def initiate_openai(ENV_File):
    _ = load_dotenv(ENV_File) # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']

    return OpenAI()