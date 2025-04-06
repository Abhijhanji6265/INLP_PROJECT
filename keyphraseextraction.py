# Keyphrase Extraction Project Implementation

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import networkx as nx
from gensim.models import KeyedVectors

# Use scikit-learn's built-in stop words
STOPWORDS = text.ENGLISH_STOP_WORDS

# Load datasets (Inspec/SemEval)
def load_dataset(path):
    df = pd.read_csv(path)
    return df

# Simple tokenizer without nltk
def simple_tokenize(text):
    return [word for word in text.lower().split() if word.isalpha() and word not in STOPWORDS]

# Preprocess documents
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special characters
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)

# TF-IDF based keyphrase extraction
def tfidf_extraction(docs, top_n=10):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(docs)
    keyphrases = []
    for i in range(X.shape[0]):
        scores = zip(vectorizer.get_feature_names_out(), X[i].toarray().flatten())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        keyphrases.append([phrase for phrase, score in sorted_scores[:top_n]])
    return keyphrases

# TextRank based keyphrase extraction
def textrank_extraction(text, top_n=10):
    words = simple_tokenize(text)
    graph = nx.Graph()
    window_size = 4
    for i in range(len(words) - window_size + 1):
        for j in range(window_size):
            for k in range(j+1, window_size):
                graph.add_edge(words[i+j], words[i+k])
    scores = nx.pagerank(graph)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [w for w, s in ranked[:top_n]]

# TopicRank (simplified)
def topicrank_extraction(text, top_n=10):
    sentences = text.split('.')
    candidates = set()
    for sentence in sentences:
        tokens = simple_tokenize(sentence)
        for i in range(len(tokens) - 1):
            candidates.add(tokens[i] + ' ' + tokens[i+1])
    graph = nx.Graph()
    for c1 in candidates:
        for c2 in candidates:
            if c1 != c2 and len(set(c1.split()).intersection(set(c2.split()))) > 0:
                graph.add_edge(c1, c2)
    scores = nx.pagerank(graph)
    return [w for w, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

# Key2Vec embedding-based ranking
def key2vec_ranking(phrases, model):
    phrase_scores = {}
    for phrase in phrases:
        words = phrase.split()
        try:
            vectors = [model[word] for word in words if word in model]
            if vectors:
                phrase_scores[phrase] = np.mean([np.linalg.norm(v) for v in vectors])
        except:
            continue
    return sorted(phrase_scores, key=phrase_scores.get, reverse=True)

# Main function to run extraction
def main():
    # Load the dataset
    dataset_path = "/kaggle/input/inspec-1/inspec.csv"  # Update with your dataset path
    df = load_dataset(dataset_path)

    # Preprocess the abstracts
    df['processed'] = df['abstract'].apply(preprocess)

    # TF-IDF Keyphrases
    tfidf_keys = tfidf_extraction(df['processed'])
    print("TF-IDF Keyphrases for first doc:", tfidf_keys[0])

    # TextRank Keyphrases
    df['textrank_keys'] = df['abstract'].apply(lambda x: textrank_extraction(x))
    print("TextRank Keyphrases for first doc:", df['textrank_keys'].iloc[0])

    # TopicRank Keyphrases
    df['topicrank_keys'] = df['abstract'].apply(lambda x: topicrank_extraction(x))
    print("TopicRank Keyphrases for first doc:", df['topicrank_keys'].iloc[0])

    # Optional: Key2Vec re-ranking using pretrained embeddings
    # Replace with actual path to pre-trained Key2Vec or Word2Vec binary model
    model_path = "/kaggle/input/key2vec-model/key2vec.bin"  # update if needed
    if os.path.exists(model_path):
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        key2vec_ranked = key2vec_ranking(tfidf_keys[0], model)
        print("Key2Vec-Ranked Keyphrases for first doc:", key2vec_ranked[:10])
    else:
        print("Key2Vec model not found. Skipping embedding-based ranking.")

if __name__ == "__main__":
    main()
