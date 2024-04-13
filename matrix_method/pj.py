import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import KeyedVectors
import csv


# calculate the similarity between two documents
def vector_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# functions to embed the text into a vector
def document_vector(text, word_vectors):
    words = text.split()
    word_vecs = [word_vectors[word] for word in words if word in word_vectors.key_to_index]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(word_vectors.vector_size)



file_path = 'data.json'
data = []
with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# for item in data:
#     print(f"ID: {item['id']}, Title: {item['title']}, Year: {item['year']}")


# CooccurrenceCnt
    
def calculate_common_citations(dict1, dict2):
    citations1 = set(dict1['citation'])
    citations2 = set(dict2['citation'])
    common_citations = citations1.intersection(citations2)
    return len(common_citations)

common_citations_count = calculate_common_citations(data[0], data[1])
print(f"Common citations: {common_citations_count}")


# Similarity EmbSimilarity TfIdfSimilarity

# content-based similarity
texts = [d["title"] + " " + d["abstract"] for d in data]


# EmbSimilarity
def calculate_emb_similarity(texts, word_vectors):
    def document_vector(doc):
        words = doc.split()
        word_vecs = [word_vectors[word] for word in words if word in word_vectors]
        return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(word_vectors.vector_size)

    doc_vectors = np.array([document_vector(text) for text in texts])
    return cosine_similarity(doc_vectors)

# load the pre-trained word vectors
word_vectors = KeyedVectors.load_word2vec_format('/Users/zyw/Documents/SI618/SI_618_WN_24_Files/data/GoogleNews-vectors-negative300-SLIM.bin', binary=True) 

# initialize the tf-idf vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)


# define the header of the csv file
headers = ['PaperId', 'RecommendedPaperId', 'Method', 'EmbSimilarity', 'TfIdfSimilarity']

# write the similarity scores to a csv file
with open('similarity_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    # write the header
    writer.writeheader()

    # loop through all pairs of papers
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            # calculate the embeddings similarity
            vec1 = document_vector(texts[i], word_vectors)
            vec2 = document_vector(texts[j], word_vectors)
            emb_similarity = vector_cosine_similarity(vec1, vec2)

            # calculate the tf-idf similarity
            tfidf_similarity = vector_cosine_similarity(tfidf_matrix[i].toarray(), tfidf_matrix[j].toarray())

            # construct the row for the csv file
            row = {
                'PaperId': data[i]['id'],
                'RecommendedPaperId': data[j]['id'],
                'Method': 'content',
                'EmbSimilarity': emb_similarity,
                'TfIdfSimilarity': tfidf_similarity
            }
            writer.writerow(row)