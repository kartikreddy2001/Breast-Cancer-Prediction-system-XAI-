import os
import random
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Main Function
# -----------------------------
def main():
    set_seed(42)

    # 🔹 LOAD CSV DATASET
    csv_path = "tumor.csv"
    df = pd.read_csv(csv_path)

    print("Dataset shape:", df.shape)
    print(df.head())

   # Drop ID column (not useful for learning)
    df = df.drop(columns=["Sample code number"])

    # Convert Class column to binary
    # 2 -> 0 (Benign), 4 -> 1 (Malignant)
    df["Class"] = df["Class"].map({2: 0, 4: 1})

    X = df.drop(columns=["Class"])
    y = df["Class"]


    X = X.values
    y = y.values

    # 🔹 TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔹 FEATURE SCALING
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    import joblib
    joblib.dump(scaler, "scaler.pkl")


    # -----------------------------
    # 🔹 DEEP NEURAL NETWORK (6 Hidden Layers)
    # -----------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation="relu"),

        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.Dense(1, activation="sigmoid")  # Binary output
    ])

    # 🔹 COMPILE MODEL
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # 🔹 CALLBACKS (BOOST ACCURACY)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ]

    # 🔹 TRAIN MODEL (≥ 30 epochs)
    history = model.fit(
        X_train,
        y_train,
        epochs=40,                 # ✅ minimum 30
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # -----------------------------
    # 🔹 EVALUATION
    # -----------------------------
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 🔹 SAVE MODEL
    model.save("deep_nn_model.keras")
    print("Model saved as deep_nn_model.keras")


if __name__ == "__main__":
    main()
