"""
train.py — Optimizer Comparison: Adam vs SGD vs RMSprop vs Adagrad
Dataset: Breast Cancer Wisconsin (from sklearn)
Task: Binary Classification using TensorFlow/Keras Neural Network
"""

import numpy as np
import time
import json
import os

import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Hyperparameters ────────────────────────────────────────────────────────
EPOCHS     = 50
BATCH_SIZE = 32
TEST_SIZE  = 0.2
LR         = 0.001   # Learning rate (same for all optimizers)


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load the Breast Cancer Wisconsin dataset from sklearn,
    apply standard scaling, and split into train/test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    data = load_breast_cancer()
    X, y = data.data, data.target          # 569 samples, 30 features

    # Standardize features: mean=0, std=1 — crucial for NNs
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reproducible split using fixed random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y        # keep class balance in both splits
    )
    print(f"[Data] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════

def build_model(optimizer, input_dim=30):
    """
    Build a simple feedforward Neural Network.
    Architecture: Input(30) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1, Sigmoid)

    Same architecture is used for EVERY optimizer (fair comparison).
    Args:
        optimizer: a compiled tf.keras.optimizers object
        input_dim: number of input features
    Returns: compiled Keras model
    """
    model = tf.keras.Sequential([
        # Hidden Layer 1: 64 neurons, ReLU activation
        tf.keras.layers.Dense(64, activation='relu',
                              input_shape=(input_dim,),
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),

        # Dropout to reduce overfitting
        tf.keras.layers.Dropout(0.2, seed=SEED),

        # Hidden Layer 2: 32 neurons, ReLU activation
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),

        # Output Layer: 1 neuron, Sigmoid → binary probability
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Binary crossentropy loss for binary classification
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ════════════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train a compiled Keras model and return the history + elapsed time.
    Args:
        model: compiled Keras model
        X_train, y_train: training data
        X_test, y_test: validation data
    Returns:
        history: Keras History object (contains loss/accuracy per epoch)
        elapsed: float, training time in seconds
    """
    start = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0       # suppress per-epoch output (clean logs)
    )

    elapsed = time.time() - start
    return history, elapsed


# ════════════════════════════════════════════════════════════════════════════
# 4. OPTIMIZERS REGISTRY
# ════════════════════════════════════════════════════════════════════════════

def get_optimizers():
    """
    Returns a dict of {name: optimizer_instance}.
    All use the same learning rate for a fair comparison.
    """
    return {
        "Adam":    tf.keras.optimizers.Adam(learning_rate=LR),
        "SGD":     tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9),
        "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=LR),
        "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=LR),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN COMPARISON LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_comparison():
    """
    Train every optimizer, collect metrics, and save results to JSON.
    Returns a dict keyed by optimizer name with all metrics.
    """
    X_train, X_test, y_train, y_test = load_data()
    optimizers = get_optimizers()
    results = {}

    for name, opt in optimizers.items():
        print(f"\n[Training] Optimizer: {name}")

        # Fresh model for each optimizer
        model = build_model(opt, input_dim=X_train.shape[1])

        history, elapsed = train_model(model, X_train, y_train, X_test, y_test)

        # Final evaluation on test set
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)
        final_loss  = history.history['val_loss'][-1]

        # Convergence speed: epoch at which val_loss first drops below threshold
        val_losses = history.history['val_loss']
        threshold  = val_losses[0] * 0.5          # 50 % of initial loss
        conv_epoch = next((i+1 for i, l in enumerate(val_losses) if l < threshold), EPOCHS)

        results[name] = {
            "accuracy":         round(test_acc * 100, 2),    # in %
            "loss":             round(final_loss, 4),
            "training_time":    round(elapsed, 2),            # in seconds
            "convergence_epoch": conv_epoch,
            "train_acc_history": [round(v, 4) for v in history.history['accuracy']],
            "val_acc_history":   [round(v, 4) for v in history.history['val_accuracy']],
            "train_loss_history":[round(v, 4) for v in history.history['loss']],
            "val_loss_history":  [round(v, 4) for v in history.history['val_loss']],
        }

        print(f"  ✓ Accuracy: {results[name]['accuracy']}% | "
              f"Loss: {results[name]['loss']} | "
              f"Time: {results[name]['training_time']}s | "
              f"Converged @ epoch {conv_epoch}")

    # Persist results so Streamlit can load them without re-training
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[Done] Results saved to results/comparison_results.json")
    return results


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_comparison()
