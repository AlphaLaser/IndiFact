from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = pd.read_csv("Data/Processed/processed_data.csv")

X = np.array(data['text'])
y = np.array(data['numeric_labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
 
np.save("Data/train_test_split/X_train.npy", X_train, allow_pickle=True)
np.save("Data/train_test_split/X_test.npy", X_test, allow_pickle=True)
np.save("Data/train_test_split/y_train.npy", y_train, allow_pickle=True)
np.save("Data/train_test_split/y_test.npy", y_test, allow_pickle=True)
