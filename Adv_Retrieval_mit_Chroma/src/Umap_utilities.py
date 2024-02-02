#import umap
import umap.umap_ as umap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

###############################################################################################
def umap_transform(chroma_collection):
    embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
    return embeddings, umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

###############################################################################################
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings  


###############################################################################################
def display_embeddings(**kwargs):
    projected_dataset_embeddings = kwargs['projected_dataset_embeddings']
    Title = kwargs['Title']

    plt.figure()
    if 'projected_augmented_query_embedding' in kwargs:
        projected_retrieved_embeddings = kwargs['projected_retrieved_embeddings']
        projected_original_query_embedding= kwargs['projected_original_query_embedding']
        projected_augmented_query_embedding = kwargs['projected_augmented_query_embedding']
        plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
        plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')
        plt.scatter(projected_original_query_embedding[:, 0], projected_original_query_embedding[:, 1], s=150, marker='X', color='r')
        plt.scatter(projected_augmented_query_embedding[:, 0], projected_augmented_query_embedding[:, 1], s=150, marker='X', color='orange')

    elif 'projected_query_embedding' in kwargs:
        projected_query_embedding = kwargs['projected_query_embedding']
        projected_retrieved_embeddings = kwargs['projected_retrieved_embeddings']
        plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
        plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=75, marker='x', color='r')
        plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')
    
    else:
        plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10)

    plt.gca().set_aspect('equal', 'datalim')
    plt.title(Title)
    plt.axis('off')
