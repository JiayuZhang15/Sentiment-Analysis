from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import review.config as config
import os


def fit_model(xtrain, xtest, ytrain, ytest, embedding, lstm_out: int, batch_size: int, epochs: int):
    """
    :param xtrain: training data set for features
    :param xtest: testing data set for features
    :param ytrain: training data set for target
    :param ytest: testing data set for target
    :param embedding: the embedding parameter of the pre-trained embedding
    :param lstm_out: the output length of the LSTM model
    :param batch_size: the size of batches
    :param epochs: how many epochs we want, even there would be a early stopping
    :return: the model
    """
    model = Sequential()  # create the model
    model.add(embedding)
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())  # print out the model details

    # create early stopping parameter
    x_train, x_validate, y_train, y_validate = train_test_split(xtrain, ytrain, test_size=0.2, random_state=20)
    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                 ModelCheckpoint(filepath=os.path.join(config.MODEL_SUBDIR, 'best_model.h5'),
                 monitor='val_loss', save_best_only=True)]

    model.fit(x_train, y_train,  # fitting the model
              callbacks=callbacks, epochs=epochs, batch_size=batch_size, verbose=2,
              validation_data=(x_validate, y_validate))

    score, acc = model.evaluate(xtest, ytest, batch_size, verbose=2)
    print('Test Score:', score, '\nTest accuracy:', acc)  # print the model result include test score and accuracy
    return model
