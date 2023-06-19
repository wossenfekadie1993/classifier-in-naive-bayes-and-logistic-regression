import math
import random
import matplotlib.pyplot as plt
import csv
from mnist import MNIST
import numpy as np
from skimage.feature import hog
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import cv2
from keras.datasets import mnist

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_counts = {}
        self.feature_counts = {}
        self.class_probabilities = {}
        self.feature_probabilities = {}
        self.classes = set()

    def train(self, X, y):
        self.classes = set(y)

        # Count occurrences of each class and each feature
        for xi, yi in zip(X, y):
            if yi not in self.class_counts:
                self.class_counts[yi] = 0
                self.feature_counts[yi] = {}
            self.class_counts[yi] += 1

            for feature in xi:
                if feature not in self.feature_counts[yi]:
                    self.feature_counts[yi][feature] = 0
                self.feature_counts[yi][feature] += 1

        # Calculate class probabilities
        total_samples = len(X)
        for c in self.classes:
            self.class_probabilities[c] = self.class_counts[c] / total_samples

        # Calculate feature probabilities
        for c in self.classes:
            total_features = sum(self.feature_counts[c].values())
            self.feature_probabilities[c] = {}
            for feature in self.feature_counts[c]:
                self.feature_probabilities[c][feature] = (
                    self.feature_counts[c][feature] + self.alpha
                ) / (total_features + self.alpha * len(self.feature_counts[c]))

    def predict(self, X):
        predictions = []
        for xi in X:
            max_prob = -math.inf
            predicted_class = None
            for c in self.classes:
                class_prob = math.log(self.class_probabilities[c])
                feature_probs = self.feature_probabilities[c]
                for feature in xi:
                    if feature in feature_probs:
                        class_prob += math.log(feature_probs[feature])
                if class_prob > max_prob:
                    max_prob = class_prob
                    predicted_class = c
            predictions.append(predicted_class)
        return predictions


# Logistic Regression Classifier
class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = []
        self.classes = set()

    def train(self, X, y, epochs=100):
        self.classes = set(y)
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(len(X[0]) + 1)]  # Add bias term

        for _ in range(epochs):
            for xi, yi in zip(X, y):
                xi = [1] + xi  # Add bias term
                predicted = self.predict_proba(xi)
                error = yi - predicted

                for j in range(len(xi)):
                    self.weights[j] += self.learning_rate * error * xi[j]

    def predict_proba(self, xi):
        z = sum(wi * xi[i] for i, wi in enumerate(self.weights))
        return 1 / (1 + math.exp(-z))

    def predict(self, X):
        predictions = []
        for xi in X:
            xi = [1] + xi  # Add bias term
            predicted = self.predict_proba(xi)
            predictions.append(1 if predicted >= 0.5 else 0)
        return predictions


# Feature extraction for MNIST dataset
def extract_raw_pixel_intensity(images):
    num_samples = images.shape[0]
    num_features = np.prod(images.shape[1:])
    reshaped_images = images.reshape(num_samples, num_features)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_images = scaler.fit_transform(reshaped_images)
    return normalized_images

def extract_hog(images):  #Histogram of Oriented Gradients(hog stands for)
    hog_features = []
    for image in images:
        hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(hog_feature)
    return np.array(hog_features)

def extract_lbp(images):  #Local Binary Patterns(lbp stands for)
    lbp_features = []
    for image in images:
        lbp_feature = local_binary_pattern(image, 8, 1, method='uniform').flatten()
        lbp_features.append(lbp_feature)
    return np.array(lbp_features)

def extract_sift(images):
    sift = cv2.SIFT_create()
    sift_features = []
    for image in images:
        _, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors = descriptors.flatten()
        else:
            descriptors = np.zeros((128,), dtype=np.float32)
        sift_features.append(descriptors)
    return np.array(sift_features)


# Feature extraction for BBC dataset

# extract Bag-of-Words (BoW) features from a list of texts 
def extract_bow_features(texts, max_features=1000):
    vectorizer = CountVectorizer(lowercase=True, max_features=max_features)
    features = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names()
    return features, vocabulary

# extract TF-IDF (Term Frequency-Inverse Document Frequency) features instead of simple Bag-of-Words features.
def extract_tfidf_features(texts, max_features=1000):
    vectorizer = TfidfVectorizer(lowercase=True, max_features=max_features)
    features = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names()
    return features, vocabulary

# extracts Word2Vec features from a list of texts
def extract_word2vec_features(texts, size=100, window=5, min_count=1, workers=4):
    tokenized_texts = [text.lower().split() for text in texts]
    model = Word2Vec(tokenized_texts, size=size, window=window, min_count=min_count, workers=workers)
    features = np.array([np.mean([model.wv[word] for word in words if word in model.wv], axis=0) for words in tokenized_texts])
    return features

# Load MNIST dataset
def load_mnist_dataset():
    mndata = MNIST('path/to/mnist_dataset')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    return train_images, train_labels, test_images, test_labels

# Load BBC dataset
def load_bbc_dataset():
    dataset_path = 'path/to/bbc_dataset.csv'
    texts = []
    labels = []
    with open(dataset_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            texts.append(row[0])
            labels.append(row[1])
    return texts, labels

# Define the hyperparameters and their possible values
hyperparameters = {
    'naive_bayes': {
        'alpha': [0.1, 0.5, 1.0, 10, 100]
    },
    'logistic_regression': {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
    }
}


# Train and test classifiers with different hyperparameters
def train_test_classifiers():
    
    # Load MNIST dataset
    train_images, train_labels, test_images, test_labels = load_mnist_dataset()

    # Feature extraction methods for MNIST dataset
    feature_extraction_methods_mnist = [
        extract_raw_pixel_intensity,extract_hog,extract_lbp,extract_sift 
        
    ]

    for method in feature_extraction_methods_mnist:
        features_train = method(train_images)
        features_test = method(test_images)

        # Train and test Naive Bayes classifier
        print("Feature Extraction Method:", method.__name__)
        print("Naive Bayes Classifier:")
        for alpha in hyperparameters['naive_bayes']['alpha']:
            nb_classifier = NaiveBayesClassifier(alpha=alpha)
            nb_classifier.train(features_train, train_labels)
            predictions = nb_classifier.predict(features_test)
            accuracy = calculate_accuracy(predictions, test_labels)
            print("Alpha:", alpha, "Accuracy:", accuracy)

        # Train and test Logistic Regression classifier
        print("Logistic Regression Classifier:")
        for learning_rate in hyperparameters['logistic_regression']['learning_rate']:
            lr_classifier = LogisticRegressionClassifier(learning_rate=learning_rate)
            lr_classifier.train(features_train, train_labels)
            predictions = lr_classifier.predict(features_test)
            accuracy = calculate_accuracy(predictions, test_labels)
            print("Learning Rate:", learning_rate, "Accuracy:", accuracy)
        print()

    # Load BBC dataset
    texts, labels = load_bbc_dataset()

    # Feature extraction methods for BBC dataset
    feature_extraction_methods_bbc = [
        extract_bow_features, extract_tfidf_features,extract_word2vec_features # Tokenized text as features
        # Add more feature extraction methods for BBC dataset
    ]

    for method in feature_extraction_methods_bbc:
        features_train, vocabulary = method(texts)
        features_test = method(test_texts)[0]  # Test feature extraction with the same method

        # Train and test Naive Bayes classifier
        print("Feature Extraction Method:", method.__name__)
        print("Naive Bayes Classifier:")
        for alpha in hyperparameters['naive_bayes']['alpha']:
            nb_classifier = NaiveBayesClassifier(alpha=alpha)
            nb_classifier.train(features_train, labels)
            predictions = nb_classifier.predict(features_test)
            accuracy = calculate_accuracy(predictions, test_labels)
            print("Alpha:", alpha, "Accuracy:", accuracy)

        # Train and test Logistic Regression classifier
        print("Logistic Regression Classifier:")
        for learning_rate in hyperparameters['logistic_regression']['learning_rate']:
            lr_classifier = LogisticRegressionClassifier(learning_rate=learning_rate)
            lr_classifier.train(features_train, labels)
            predictions = lr_classifier.predict(features_test)
            accuracy = calculate_accuracy(predictions, test_labels)
            print("Learning Rate:", learning_rate, "Accuracy:", accuracy)
        print()


# Calculate accuracy
def calculate_accuracy(predictions, labels):
    correct_count = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct_count / len(labels)


# Plot accuracy change based on hyperparameter variations
def plot_accuracy_change(x_values, y_values, x_label, y_label, title):
    plt.plot(x_values, y_values, 'o-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


# Example usage
train_test_classifiers()

def load_mnist_dataset():
    (train_images_path,train_labels_path),(test_images_path,test_labels_path) = mnist.load_data()
    # train_labels_path = 'C:\Users\Wondalem\Desktop\classifier\train-labels.idx1-ubyte'
    # test_images_path = 'C:\Users\Wondalem\Desktop\classifier\t10k-images.idx3-ubyte'
    # test_labels_path = 'C:\Users\Wondalem\Desktop\classifier\t10k-labels.idx1-ubyte'

    # mndata = MNIST('C:\Users\Wondalem\Documents\ai for final\python-mnist-master\python-mnist-master\bin')
    # train_images, train_labels = mndata.load_training(train_images_path, train_labels_path)
    # test_images, test_labels = mndata.load_testing(test_images_path, test_labels_path)

    return train_images_path, train_labels_path, test_images_path, test_labels_path
