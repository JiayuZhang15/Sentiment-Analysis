from review.utils import preprocess, model, embedding
import review.config as config
import os.path


def sentiment_analysis(file1, file2, embedding_file):
    """
    This is the collection of all defined functions, it will process the data and train the model
    :param file1: the train data location
    :param file2: the test data location
    :param embedding_file: the file contains pre-trained embedding
    :return: the final result
    """
    # call the pre-process
    x_train, x_test, y_train, y_test, word_index, xte = preprocess.text_process(
        file_path1=os.path.join(config.DATA_SUBDIR, file1),
        file_path2=os.path.join(config.DATA_SUBDIR, file2),
        text_data='Phrase',
        label='Sentiment',
        features=10000)

    # call embedding
    embeddings = embedding.embedding(embedding_file, word_index=word_index, xtrain=x_train)
    # call model
    result = model.fit_model(x_train, x_test, y_train, y_test, embeddings,
                             lstm_out=196, batch_size=32, epochs=12)
    print(result)
    preprocess.save_binary(xte, os.path.join(config.DATA_SUBDIR, 'xtest2'))
    print('X from test data is ready for predict')


if __name__ == '__main__':
    sentiment_analysis('train.tsv', 'test.tsv', 'glove.42B.300d.txt')
