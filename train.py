import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from keras.callbacks import ModelCheckpoint
import os

# --- CONFIG ---
DATA_DIR = "UIPRMDsquat/fold0"


# --- NORMALIZATION FUNCTION (The Secret Sauce) ---
def normalize_skeleton(data):
    """
    Input: (Batch, Time, Joints*3)
    1. Reshape to (Batch, Time, Joints, 3)
    2. Center Hip to (0,0,0)
    3. Scale so Spine Length = 1.0
    4. Flatten back
    """
    # 1. Reshape
    B, T, F = data.shape
    J = F // 3
    data = data.reshape(B, T, J, 3)

    # 2. Center at Mid-Hip (Joint 0 in our mapping)
    # We assume Joint 0 is the root/hip based on our mapping
    root = data[:, :, 0:1, :]  # (B, T, 1, 3)
    data = data - root

    # 3. Scale by Spine Length (Distance between Hip(0) and Neck(2)/Shoulder(2))
    # Using Joint 2 (SpineShoulder) as the top anchor
    spine_top = data[:, :, 2:3, :]
    spine_len = np.linalg.norm(spine_top, axis=3, keepdims=True)

    # Avoid division by zero
    spine_len = np.maximum(spine_len, 0.0001)

    data = data / spine_len

    # 4. Flatten
    return data.reshape(B, T, F)


# --- LOAD & PROCESS ---
def load_data(fold_path):
    x_train = np.load(os.path.join(fold_path, "x_train_fold0.npy"))
    y_train = np.load(os.path.join(fold_path, "y_train_fold0.npy"))
    x_test = np.load(os.path.join(fold_path, "x_test_fold0.npy"))
    y_test = np.load(os.path.join(fold_path, "y_test_fold0.npy"))
    return x_train, y_train, x_test, y_test


print("Loading data...")
x_train, y_train, x_test, y_test = load_data(DATA_DIR)

# Transpose if needed (Features last)
if x_train.shape[1] == 66:
    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)

print("Normalizing Training Data...")
# We normalize Train and Test separately to avoid data leakage,
# but using the same logic.
x_train = normalize_skeleton(x_train)
x_test = normalize_skeleton(x_test)

# --- TRAIN MODEL ---
model = Sequential([
    Input(shape=(81, 66)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print("Starting Robust Training...")
checkpoint_path = os.path.join(os.path.dirname(__file__), 'deep_squat_robust.keras')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_mae', save_best_only=True, mode='min')

model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint]
)

print("Training Complete. Model saved as 'deep_squat_robust.keras'")