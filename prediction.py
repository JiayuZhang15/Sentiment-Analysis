from keras import models
import review.config as config
from review.utils.preprocess import save_csv, load_binary
import pandas as pd
import os

'''
def predict(xfile, file_name, model_name):
    best_model = models.load_model(os.path.join(config.MODEL_SUBDIR, model_name))
    best_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    x_test = load_binary(os.path.join(config.DATA_SUBDIR, xfile))
    file2 = pd.read_csv(os.path.join(config.DATA_SUBDIR, file_name), sep='\t')
    y_new = best_model.predict(x_test)
    result['prediction'] = result[1]*1+result[2]*2+result[3]*3+result[4]*4
predicts = pd.DataFrame({'Phrase': file2['Phrase'], 'y': result['prediction']})
    return predicts


if __name__ == '_main_':
    prediction = predict('xtest', 'test.tsv', 'best_model.h5')
    save_csv(prediction, 'prediction.csv')
'''

best_model = models.load_model(os.path.join(config.MODEL_SUBDIR, 'best_model.h5'))
best_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
x_test = load_binary(os.path.join(config.DATA_SUBDIR, 'xtest'))
file2 = pd.read_csv(os.path.join(config.DATA_SUBDIR, 'test.tsv'), sep='\t')
y_new = best_model.predict(x_test)
result = pd.DataFrame(y_new)
print(result)
