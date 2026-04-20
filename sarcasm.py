import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from clean import limpiar_texto

df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)
print(df.head())

print(df.shape, df.columns)

print("\n---")
print(df.isnull().sum())

print("\n---")
print(df["is_sarcastic"].value_counts())

X = df["headline"]
y = df["is_sarcastic"]

X_limpio = X.apply(limpiar_texto)

for i in range(5):
    print("original:", X.iloc[i])
    print("limpio  :", X_limpio.iloc[i])
    print()

X_temp, X_test, y_temp, y_test = train_test_split(
    X_limpio,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.125,
    random_state=42,
    stratify=y_temp
)

print("\n---")
print("train:", X_train.shape[0])
print("validation:", X_val.shape[0])
print("test:", X_test.shape[0])

max_tokens = 10000
sequence_length = 30

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)

vectorizer.adapt(X_train.to_numpy())

X_train_vec = vectorizer(np.array(X_train)).numpy()
X_val_vec = vectorizer(np.array(X_val)).numpy()
X_test_vec = vectorizer(np.array(X_test)).numpy()

print("\n---")
print("X_train vectorized:", X_train_vec.shape)
print("X_val vectorized:", X_val_vec.shape)
print("X_test vectorized:", X_test_vec.shape)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length,)),
    tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\n---")
print("entrenando modelo deep learning...")

history = modelo.fit(
    X_train_vec,
    y_train,
    validation_data=(X_val_vec, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\n---")
print("evaluando en validation")

y_val_prob = modelo.predict(X_val_vec)
y_val_pred = (y_val_prob >= 0.5).astype(int).ravel()

print("accuracy:", accuracy_score(y_val, y_val_pred))
print("precision:", precision_score(y_val, y_val_pred))
print("recall:", recall_score(y_val, y_val_pred))
print("F1-score:", f1_score(y_val, y_val_pred))

print("\nREPORTE DE CLASIFICACIÓN VALIDATION")
print(classification_report(y_val, y_val_pred))

print("\nMATRIZ DE CONFUSIÓN VALIDATION")
print(confusion_matrix(y_val, y_val_pred))

print("\n---")
print("evaluando en test")

y_test_prob = modelo.predict(X_test_vec)
y_test_pred = (y_test_prob >= 0.5).astype(int).ravel()

print("accuracy:", accuracy_score(y_test, y_test_pred))
print("precision:", precision_score(y_test, y_test_pred))
print("recall:", recall_score(y_test, y_test_pred))
print("F1-score:", f1_score(y_test, y_test_pred))

print("\nREPORTE DE CLASIFICACIÓN TEST")
print(classification_report(y_test, y_test_pred))

print("\nMATRIZ DE CONFUSIÓN TEST")
print(confusion_matrix(y_test, y_test_pred))

modelo.save("modelo_dl.keras")

vocabulario = vectorizer.get_vocabulary()
with open("vocabulario.txt", "w", encoding="utf-8") as f:
    for token in vocabulario:
        f.write(token + "\n")

print("\nmodelo y vocabulario guardados")