import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow_hub as hub
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

X_train = np.load("Data/train_test_split/X_train.npy", allow_pickle=True)
X_test = np.load("Data/train_test_split/X_test.npy", allow_pickle=True)
y_test = np.load("Data/train_test_split/y_test.npy", allow_pickle=True)
y_train = np.load("Data/train_test_split/y_train.npy", allow_pickle=True)

def compile_model(model):
    '''
    simply compile the model with adam optimzer
    '''
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])    

def fit_model(model, epochs, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    '''
    fit the model with given epochs, train and test data
    '''
    history = model.fit(X_train,
              y_train,
             epochs=epochs,
             validation_data=(X_test, y_test),
             validation_steps=int(0.2*len(X_test)))
    return history

def evaluate_model(model, X, y):
    '''
    evaluate the model and returns accuracy, precision, recall and f1-score 
    '''
    y_preds = np.round(model.predict(X))
    accuracy = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)
    
    model_results_dict = {'accuracy':accuracy,
                         'precision':precision,
                         'recall':recall,
                         'f1-score':f1}
    
    return model_results_dict



# model with Sequential api
model = keras.Sequential()
# universal-sentence-encoder layer directly from tfhub
use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                           trainable=False,
                           input_shape=[],
                           dtype=tf.string,
                           name='USE')
model.add(use_layer)
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation=keras.activations.relu))
model.add(layers.Dense(1, activation=keras.activations.sigmoid))

compile_model(model)

history = fit_model(model, epochs=10)

print(evaluate_model(model, X_test, y_test))