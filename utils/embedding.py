from keras.layers import Embedding
import review.config as config
import numpy as np
import os


def embedding(glove, word_index, xtrain):
    """
    :param glove: the file name of the pre-trained embedding GloVe file
    :param word_index: the maximum length of the text column
    :param xtrain: training set, the only purpose of this is to get X.shape[1] for input_length
    :return: embedding_layer, which is weighted embedding ready for use
    """
    embedding_dim = int(glove[-8:-5])
    embeddings_index = {}
    f = open(os.path.join(config.GLOVE_SUBDIR, glove))
    # load the glove file and create embedding index
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Embedding: Found %s word vectors.' % len(embeddings_index))

    # now we need to create embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # This would be the embedding used in the model
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=xtrain.shape[1],
                                trainable=False)
    return embedding_layer
