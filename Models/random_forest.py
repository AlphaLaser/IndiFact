from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_train = np.load("Data/train_test_split/X_train.npy", allow_pickle=True)
X_test = np.load("Data/train_test_split/X_test.npy", allow_pickle=True)
y_train = np.load("Data/train_test_split/y_train.npy", allow_pickle=True)
y_test = np.load("Data/train_test_split/y_test.npy", allow_pickle=True)

tfidf = TfidfVectorizer().fit(X_train)

X_train_vect = tfidf.transform(X_train)
X_test_vect = tfidf.transform(X_test)

rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train_vect, y_train)

from sklearn.metrics import classification_report, accuracy_score

random_forest_accuracy = accuracy_score(y_test, rfc.predict(X_test_vect))
print(classification_report(y_test, rfc.predict(X_test_vect)))