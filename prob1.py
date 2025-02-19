import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

def load_data():
    # Load the 20 Newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    X, y = newsgroups.data, newsgroups.target
    return X, y, newsgroups.target_names

def preprocess_data(X_train, X_test, min_occurrences=1000):
    # Convert documents to bag-of-words representation
    vectorizer = CountVectorizer(max_features=60000)
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    # Compute word frequencies in training set
    word_counts = np.array(X_train_counts.sum(axis=0)).flatten()
    frequent_words = np.where(word_counts > min_occurrences)[0]
    
    # Reduce feature space by keeping only frequent words
    X_train_pruned = X_train_counts[:, frequent_words]
    X_test_pruned = X_test_counts[:, frequent_words]
    
    return X_train_pruned, X_test_pruned

def estimate_probabilities(X_train, y_train, num_classes):
    # Compute class priors P(y=c)
    class_counts = np.bincount(y_train)
    pi_c = class_counts / class_counts.sum()
    
    # Compute P(x|y) (multinomial word probabilities per class)
    vocab_size = X_train.shape[1]
    pc = np.zeros((num_classes, vocab_size))
    
    for c in range(num_classes):
        class_docs = X_train[y_train == c]
        word_counts = np.array(class_docs.sum(axis=0)).flatten()
        pc[c] = (word_counts + 1) / (word_counts.sum() + vocab_size)  # Laplace smoothing
    
    return pc, pi_c

def train_multinomial_nb(X_train, y_train, num_classes):
    pc, pi_c = estimate_probabilities(X_train, y_train, num_classes)
    wc = np.log(pc)  # Compute log probabilities
    beta_c = np.log(pi_c)
    return wc, beta_c

def predict(X, wc, beta_c):
    # Compute log-probability for each class
    log_probs = X @ wc.T + beta_c
    return np.argmax(log_probs, axis=1)

# Load and split dataset
X, y, class_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess data
X_train_pruned, X_test_pruned = preprocess_data(X_train, X_test)

# Train Naïve Bayes classifier
num_classes = len(class_names)
wc, beta_c = train_multinomial_nb(X_train_pruned, y_train, num_classes)

# Make predictions
y_pred = predict(X_test_pruned, wc, beta_c)

# Compute accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Prediction Accuracy: {accuracy * 100:.2f}%")
