from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import review.config as config
import pandas as pd
import pickle
import re
import os


def text_process(file_path1, file_path2, text_data: str, label: str, features: int):
    """
    :param file_path1: directory of training data
    :param file_path2: directory of testing data
    :param text_data: the name of column that store the text data
    :param label: the name of target column
    :param features: number of max feature, which is the count of max word
    :return: x_train, x_test, y_train, y_test, word_index
    """
    data1 = pd.read_csv(file_path1, sep='\t')  # Training data
    data2 = pd.read_csv(file_path2, sep='\t')  # Testing data
    data3 = data1[[text_data, label]]  # Retain the training data science we need the label
    seed = len(data1)  # set a seed at the end of training data, so we can split them after tokenizer
    data1 = data1[[text_data]]
    data2 = data2[[text_data]]
    data = data1.append(data2)

    data[text_data] = data[text_data].apply(lambda xt: xt.lower())
    data[text_data] = data[text_data].apply((lambda xt: re.sub('[^a-zA-z0-9\s]', '', xt)))
#   keep only words and numbers, remove all special characters

    max_fatures = features
    tokenizer = Tokenizer(num_words=max_fatures)
    tokenizer.fit_on_texts(data[text_data].values)
    word_index = tokenizer.word_index
    x = tokenizer.texts_to_sequences(data[text_data].values)
    x = sequence.pad_sequences(x)
    y = pd.get_dummies(data3[label]).values  # use the retain data to get label
    xtr = x[0:seed]  # use the seed we set to get back the training data of X
    xte = x[seed:]
    x_train, x_test, y_train, y_test = train_test_split(xtr, y, test_size=0.2, random_state=20)
    print('xtrain shape is:', x_train.shape, 'ytrain shape is:', y_train.shape)
    print('xtest shape is:', x_test.shape, 'ytest shape is:', y_test.shape)
    return x_train, x_test, y_train, y_test, word_index, xte


"""
Below are spare functions for saving and loading
"""


def save_binary(obj, path):
    pickle.dump(obj, open(path, 'wb'))
    print('File Saved')


def load_binary(path):
    print('File Loaded')
    return pickle.load(open(path, 'rb'))


def save_csv(file_name, save_name):
    path = os.path.join(config.DATA_SUBDIR, save_name)
    file_name.to_csv(path, sep=',')
    print('file exported')
