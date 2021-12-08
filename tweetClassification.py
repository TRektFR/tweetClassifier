import numpy as np
import joblib
from stop_words import get_stop_words
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import html

com = pd.read_csv("labels.csv")

com["tweet"] = com["tweet"].transform(lambda x: html.unescape(x))

X = com["tweet"]
y = com[["class"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=102)

clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('english')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True)),
)

clf.fit(X_train,y_train)

print(clf.score(X_test, y_test))

joblib.dump(clf,"model.pkl")
