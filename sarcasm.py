import json
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from clean import limpiar_texto

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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

max_tokens = 8000
sequence_length = 25 

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
    tf.keras.layers.Embedding(input_dim=max_tokens, output_dim= 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR")
    ]
)

modelo.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2, 
    restore_best_weights=True
)

print("\n---")
print("entrenando...")

history = modelo.fit(
    X_train_vec,
    y_train,
    validation_data=(X_val_vec, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

history_dict = history.history

with open("training_history.json", "w", encoding="utf-8") as f:
    json.dump(history_dict, f, ensure_ascii=False, indent=4)

best_epoch = np.argmin(history.history["val_loss"])

print("\n---")
print("mejor época:", best_epoch + 1)
print("mejor val_loss:", history.history["val_loss"][best_epoch])
print("val_roc_auc:", history.history["val_roc_auc"][best_epoch])
print("val_pr_auc :", history.history["val_pr_auc"][best_epoch])

print("\n---")
print("evaluando en validation")

y_val_prob = modelo.predict(X_val_vec).ravel()
y_test_prob = modelo.predict(X_test_vec).ravel()

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

best_threshold = None
best_f1 = -1

for threshold in thresholds:
    y_val_pred = (y_val_prob >= threshold).astype(int)

    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred)
    rec = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    print(f"\nthreshold: {threshold}")
    print("accuracy :", acc)
    print("precision:", prec)
    print("recall   :", rec)
    print("f1-score :", f1)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\n---")
print("mejor threshold en validation:", best_threshold)
print("mejor f1 en validation:", best_f1)

y_val_pred_best = (y_val_prob >= best_threshold).astype(int)
y_test_pred = (y_test_prob >= best_threshold).astype(int)

best_val_metrics = {
    "accuracy": accuracy_score(y_val, y_val_pred_best),
    "precision": precision_score(y_val, y_val_pred_best),
    "recall": recall_score(y_val, y_val_pred_best),
    "f1_score": f1_score(y_val, y_val_pred_best)
}

print("\nREPORTE DE CLASIFICACIÓN VALIDATION")
print(classification_report(y_val, y_val_pred_best))

print("\nMATRIZ DE CONFUSIÓN VALIDATION")
print(confusion_matrix(y_val, y_val_pred_best))

print("\n---")
print("evaluando en test con el mejor threshold")

print("accuracy :", accuracy_score(y_test, y_test_pred))
print("precision:", precision_score(y_test, y_test_pred))
print("recall   :", recall_score(y_test, y_test_pred))
print("f1-score :", f1_score(y_test, y_test_pred))

print("\nREPORTE DE CLASIFICACIÓN TEST")
print(classification_report(y_test, y_test_pred))

print("\nMATRIZ DE CONFUSIÓN TEST")
print(confusion_matrix(y_test, y_test_pred))

modelo.save("modelo_dl.keras")

vocabulario = vectorizer.get_vocabulary()

metadata = {
    "best_threshold": best_threshold,
    "max_tokens": max_tokens,
    "sequence_length": sequence_length,
    "best_validation_metrics": best_val_metrics,
    "vocabulario": vocabulario
}

with open("preprocessing_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("\nmodelo, vocabulario y threshold guardados")