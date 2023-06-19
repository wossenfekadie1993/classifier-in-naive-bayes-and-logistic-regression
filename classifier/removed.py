# def extract_features_bbc(texts):
#     features = []
#     vocabulary = set()
#     for text in texts:
#         tokens = text.lower().split()
#         features.append(tokens)
#         vocabulary.update(tokens)
#     vocabulary = list(vocabulary)
#     return features, vocabulary




# def extract_features_mnist(images):
#     features = []
#     for image in images:
#         features.append(image.flatten())
#     return features