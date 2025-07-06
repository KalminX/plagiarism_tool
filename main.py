import os
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data if not already downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# Text Preprocessing Function
# ----------------------------
def preprocess(text):
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    tokens = [w.translate(table) for w in tokens if w.isalpha() and w not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)

# ----------------------------
# Load and Preprocess Documents
# ----------------------------
def load_documents(example_path, reference_paths):
    with open(example_path, 'r', encoding='utf-8') as f:
        example_text = f.read()

    reference_texts = []
    for path in reference_paths:
        with open(path, 'r', encoding='utf-8') as f:
            reference_texts.append(f.read())

    preprocessed_docs = [preprocess(example_text)] + [preprocess(text) for text in reference_texts]
    return preprocessed_docs

# ----------------------------
# Plagiarism Detection
# ----------------------------
def detect_plagiarism(docs, threshold=0.6):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)

    query_vec = tfidf_matrix[0]
    reference_vecs = tfidf_matrix[1:]

    similarities = cosine_similarity(query_vec, reference_vecs)

    print(f"\nTF-IDF Features: {tfidf_matrix.shape[1]}")
    print(f"Query Vector Shape: {query_vec.shape}")
    print(f"Reference Vector Shape: {reference_vecs.shape}")
    print(f"Similarity Scores: {similarities}")

    for idx, score in enumerate(similarities[0]):
        if score >= threshold:
            print(f"\n⚠️  Plagiarism detected with reference{idx + 1}.txt (score: {score:.3f})")
        else:
            print(f"\n✅ No plagiarism detected with reference{idx + 1}.txt (score: {score:.3f})")

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    # Setup file paths
    example_doc = 'example_document.txt'
    reference_docs = [f for f in os.listdir() if f.startswith('reference') and f.endswith('.txt')]

    if not os.path.exists(example_doc) or not reference_docs:
        print("❌ Missing files. Ensure 'example_document.txt' and at least one 'reference*.txt' are in the same directory.")
    else:
        print(f"🔍 Checking {example_doc} against {len(reference_docs)} reference documents...\n")
        documents = load_documents(example_doc, reference_docs)
        detect_plagiarism(documents)

