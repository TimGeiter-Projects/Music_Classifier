import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# ----------------------------
# Konfiguration
# ----------------------------
DATA_DIR    = r"C:\Users\geite\Desktop\Uni\Master\WS25_26\Maschinelles_Lernen\music_classifier\Raw_ML"
OUTPUT_DIR  = DATA_DIR
RANDOM_SEED = 42
BATCH_SIZE  = 32
EPOCHS      = 100

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------
# 1. Daten laden
# ----------------------------
print("=" * 60)
print("CNN Training mit Mel-Spektrogrammen (5s Segmente)")
print("=" * 60)

print("\n[1/3] Lade Daten...")
X_trainval = np.load(os.path.join(DATA_DIR, "X.npy"))
y_trainval = np.load(os.path.join(DATA_DIR, "y.npy"))
genres     = np.load(os.path.join(DATA_DIR, "genres.npy"))

print(f"   ✓ Train+Val: {X_trainval.shape[0]} Segmente")
print(f"   ✓ Genres ({len(genres)}): {', '.join(genres)}")

# ----------------------------
# 2. Train/Val Split
# ----------------------------
print("\n[2/3] Splitte Train+Val in Train (89%) und Val (11%)...")

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.111,
    random_state=RANDOM_SEED,
    stratify=y_trainval
)

print(f"   ✓ Train: {X_train.shape[0]} Segmente")
print(f"   ✓ Val:   {X_val.shape[0]} Segmente")

# ----------------------------
# 3. Modell
# ----------------------------
def build_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="MusicCNN")

# ----------------------------
# 4. Training
# ----------------------------
print(f"\n[3/3] Training...")

model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(genres))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb_list = [
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss', patience=20,
        restore_best_weights=True, verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    verbose=1
)

_, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Accuracy: {val_acc:.2%}")

# ----------------------------
# 5. Training Kurven
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History – 5s Segmente', fontsize=13)

axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.show()
print("   ✓ Training Kurven gespeichert")

# ----------------------------
# 6. Modell speichern
# ----------------------------
model.save(os.path.join(OUTPUT_DIR, "best_model.keras"))
print(f"   ✓ Modell gespeichert: best_model.keras")