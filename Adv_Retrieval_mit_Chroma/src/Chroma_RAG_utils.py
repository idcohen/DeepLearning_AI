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
from dotenv import load_dotenv
import matplotlib as plt

###############################################################################################
def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)
    
###############################################################################################
def read_pdf(filename):
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    # Filter the empty strings - exclude empty pages
    return [text for text in pdf_texts if text]

###############################################################################################
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

###############################################################################################
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

###############################################################################################
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

###############################################################################################
def initiate_openai(ENV_File):
    _ = load_dotenv(ENV_File) # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']
    return OpenAI()


###############################################################################################
def Query_Chroma(chroma_collection, query, n_results = 5, verbose=False):
    results = chroma_collection.query(query_texts=[query], n_results=n_results, include=['documents', 'embeddings'])
    retrieved_documents = results['documents'][0]
    retrieved_embeddings = results['embeddings'][0]
    if verbose:
        for i, document in enumerate(results['documents'][0]):
            print(f'{i}: {word_wrap(document)}')
            print('')
    return retrieved_documents, retrieved_embeddings
###############################################################################################
def RAG(openai_client, query, retrieved_documents, model="gpt-3.5-turbo"):
    # join retrieved documents with "\n\n"
    information = "\n\n".join(retrieved_documents)

    # system prompt - instructs LLM to transform from a model that remembers facts to one that 
    #                   processes information
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


###############################################################################################
def augment_query_generated(openai_client, query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report. "
        },
        {"role": "user", "content": query}
    ] 

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

###############################################################################################
def Query_Expansion_augment_query(openai_client, original_query, model="gpt-3.5-turbo", verbose=False):
    hypothetical_answer = augment_query_generated(openai_client, original_query, model)
    joint_query = f"{original_query} {hypothetical_answer}"
    if verbose:
            print(word_wrap(joint_query))
    return joint_query

###############################################################################################
def augment_multiple_query(openai_client, query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about an annual report. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

###############################################################################################
def Query_Expansion_Multiple_Queries(openai_client, original_query, model="gpt-3.5-turbo", verbose=False):
    augmented_queries = augment_multiple_query(openai_client, original_query, model)
    queries = [original_query] + augmented_queries
    if verbose:
            print(word_wrap(queries))
    return augmented_queries, queries
    