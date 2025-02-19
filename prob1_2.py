import os
import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import nltk
import ssl

# Fix NLTK SSL error (if needed)
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

def load_data(base_folder):
    """
    Loads the 20 Newsgroups dataset from the specified folder,
    tokenizes text, and returns document texts with labels.
    """
    categories = [
        'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
        'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
        'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
    ]
    
    data, labels = [], []

    for label, category in enumerate(categories):
        category_path = os.path.join(base_folder, category)
        if not os.path.exists(category_path):
            warnings.warn(f"Category folder {category_path} not found, skipping.")
            continue

        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    data.append(content)
                    labels.append(category)
            except Exception as e:
                warnings.warn(f"Error reading {file_path}: {str(e)}")

    return data, labels

# Load training and test data
train_folder = "20news-bydate/20news-bydate-train"
test_folder = "20news-bydate/20news-bydate-test"

X_train, y_train = load_data(train_folder)
X_test, y_test = load_data(test_folder)

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Create a bag-of-words model with word pruning (keep words appearing >1000 times)
vectorizer = CountVectorizer(min_df=1000)  # Vocabulary reduction
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

words = vectorizer.get_feature_names_out()

# Compute class priors p(y)
num_classes = len(set(y_train))
class_counts = np.bincount(y_train)
p_y = class_counts / np.sum(class_counts)  # Prior probabilities of classes

# Compute word likelihoods p(x|y) with Laplace smoothing
alpha = 1  # Laplace smoothing
vocab_size = X_train_bow.shape[1]

p_x_given_y = np.zeros((num_classes, vocab_size))

for c in range(num_classes):
    class_docs = X_train_bow[np.where(y_train == c)]
    word_counts = class_docs.sum(axis=0)  # Count words in class c
    total_words = np.sum(word_counts)  # Total words in class c

    p_x_given_y[c, :] = (word_counts + alpha) / (total_words + vocab_size * alpha)

# Convert probabilities to log probabilities to prevent underflow
w_c = np.log(p_x_given_y)  # log(pcj)
beta_c = np.log(p_y)  # log(πc)

# Classify test documents using log probabilities
scores = X_test_bow @ w_c.T + beta_c
y_pred = np.argmax(scores, axis=1)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
