import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
X_test     = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test     = np.load(os.path.join(DATA_DIR, "y_test.npy"))
genres     = np.load(os.path.join(DATA_DIR, "genres.npy"))

print(f"   ✓ Train+Val: {X_trainval.shape[0]} Segmente")
print(f"   ✓ Test:      {X_test.shape[0]} Segmente")
print(f"   ✓ Genres ({len(genres)}): {', '.join(genres)}")

# ----------------------------
# 2. Train/Val Split (aus Train+Val)
# Ergibt effektiv: 80% Train / 10% Val / 10% Test
# ----------------------------
print("\n[2/3] Splitte Train+Val in Train (89%) und Val (11%)...")
print("   → Ergibt Gesamt-Split: ~80% Train / 10% Val / 10% Test")

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.111,  # 10% von 90% = 11.1% vom Gesamt
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

# ----------------------------
# 5. Evaluation auf Val Set
# ----------------------------
_, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n{'=' * 60}")
print(f"VALIDATION ACCURACY: {val_acc:.2%}")
print(f"{'=' * 60}")

# ----------------------------
# 6. Evaluation auf Test Set
# ----------------------------
print("\nEvaluierung auf Testset...")
y_pred   = np.argmax(model.predict(X_test), axis=1)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 60}")
print(f"FINAL TEST ACCURACY: {test_acc:.2%}")
print(f"{'=' * 60}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=genres))

print("\n--- Accuracy pro Genre ---")
for i, genre in enumerate(genres):
    mask = y_test == i
    if mask.sum() > 0:
        genre_acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"   {genre}: {genre_acc:.2%}")

# Modell speichern
model.save(os.path.join(OUTPUT_DIR, "best_model.keras"))
print(f"\n   ✓ Modell gespeichert als: best_model.keras")

# ----------------------------
# 7. Training Kurven
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
print("   ✓ Training Kurven gespeichert als: training_curves.png")

# ----------------------------
# 8. Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=genres, yticklabels=genres)
plt.ylabel('Wahre Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.title(f'Confusion Matrix – CNN Mel-Spektrogramme (Acc: {test_acc:.2%})')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_cnn.png"), dpi=300)
plt.show()
print("   ✓ Confusion Matrix gespeichert als: confusion_matrix_cnn.png")

# ----------------------------
# Zusammenfassung
# ----------------------------
print(f"\n{'=' * 60}")
print("Training abgeschlossen!")
print(f"   Val Accuracy:  {val_acc:.2%}")
print(f"   Test Accuracy: {test_acc:.2%}")
print(f"   Modell:        best_model.keras")
print(f"{'=' * 60}")