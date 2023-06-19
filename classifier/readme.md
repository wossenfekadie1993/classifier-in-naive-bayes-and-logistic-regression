wossen fekadie UGR/1993/13
yabsra yetwale UGR/8664/13
seblewongel takele UGR/9711/13
Lidya Ejigu UGR/5038/13
SAMUEL ZEBENE UGR/2139/13

THE REASON WHY WE CHOOSE THE FEATURE EXTRACTION METHOD

three feature extraction methods for the MNIST and BBC datasets.

1. MNIST Dataset:
   a. Pixel Intensity: This method represents each image in the MNIST dataset as a vector of pixel intensities. Each pixel value is treated as a feature, and the intensity value is normalized between 0 and 1. This method works well for the MNIST dataset as it captures the spatial information present in the images.

   b. Histogram of Oriented Gradients (HOG): HOG is a popular feature extraction method used for object detection and recognition tasks. It calculates the gradients of image patches and constructs a histogram of gradient orientations. HOG captures shape and edge information, which can be useful for distinguishing different digits in the MNIST dataset.

   c. Principal Component Analysis (PCA): PCA is a dimensionality reduction technique that can be applied to the MNIST dataset. It identifies the most significant patterns or features in the data and projects it onto a lower-dimensional space. PCA is useful for reducing the dimensionality of the MNIST dataset while preserving the most important information.

2. BBC Dataset:

   a. Bag-of-Words (BoW): BoW is a common technique used to represent each document as a histogram of word frequencies. The idea is to create a vocabulary of unique words across the dataset and count the occurrences of these words in each document. BoW captures the distributional information of words and can be effective in text classification tasks.

   b. Term Frequency-Inverse Document Frequency (TF-IDF): TF-IDF is another popular method used to measures the importance of a term in a document relative to the entire corpus. It combines the term frequency (TF) and inverse document frequency (IDF) to assign weights to words. TF-IDF gives higher weights to terms that are frequent in a document but rare in the entire corpus, capturing the discriminative power of words.

   c. Word Embeddings (e.g., Word2Vec or GloVe): Word embeddings are dense vector representations of words that capture semantic and syntactic relationships between them.  used to convert words into fixed-length vectors. Word embeddings are effective in capturing semantic meaning and context in text data.

These feature extraction methods were chosen based on their effectiveness in capturing relevant information from the respective datasets. so we  try to implement the classifiers using Naive Bayes and Logistic Regression using those feature extractions ways.