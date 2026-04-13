# run: python sarcasm.py

import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)
print(df.head())

print(df.shape, df.columns)

print("\n---")
print(df.isnull().sum())

print("\n---")
print(df["is_sarcastic"].value_counts())

X = df["headline"]
y = df["is_sarcastic"]

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"\d+", "", texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = texto.strip()
    return texto

X_limpio = X.apply(limpiar_texto)

for i in range(5):
    print("original:", X.iloc[i])
    print("limpio  :", X_limpio.iloc[i])
    print()

X_train, X_test, y_train, y_test = train_test_split(
    X_limpio,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n---")
print("train:", X_train.shape[0])
print("test:", X_test.shape[0])

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print("\n---")
print("train shape:", X_train_tfidf.shape)
print("test shape :", X_test_tfidf.shape)

scaler = MaxAbsScaler()

X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled  = scaler.transform(X_test_tfidf)
# print("escalamiento listo")

print("\n---")
print("X_train final:", X_train_scaled.shape)
print("X_test final :", X_test_scaled.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
