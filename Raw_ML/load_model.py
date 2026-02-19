import numpy as np
import os
from sklearn.metrics import accuracy_score
import tensorflow as tf

# ----------------------------
# Konfiguration
# ----------------------------
DATA_DIR = r"C:\Users\geite\Desktop\Uni\Master\WS25_26\Maschinelles_Lernen\music_classifier\Raw_ML"

print("=" * 60)
print("Schnellcheck: Gespeichertes Modell validieren")
print("=" * 60)

# Daten laden
print("\nLade Daten...")
X_trainval = np.load(os.path.join(DATA_DIR, "X.npy"))
y_trainval = np.load(os.path.join(DATA_DIR, "y.npy"))
X_test     = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test     = np.load(os.path.join(DATA_DIR, "y_test.npy"))
genres     = np.load(os.path.join(DATA_DIR, "genres.npy"))

# Val-Set rekonstruieren (gleicher Split wie im Training)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.111,
    random_state=42,
    stratify=y_trainval
)

print(f"   ✓ Val Set:  {X_val.shape[0]} Segmente")
print(f"   ✓ Test Set: {X_test.shape[0]} Segmente")

# Modell laden
print("\nLade Modell...")
model = tf.keras.models.load_model(os.path.join(DATA_DIR, "best_model.keras"))
print("   ✓ Modell geladen: best_model.keras")

# Validation Set
print("\n--- VALIDATION SET ---")
y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

# Test Set
print("\n--- TEST SET ---")
y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")

# Genre-weise Accuracy auf Test
print("\n--- Test Accuracy pro Genre ---")
for i, genre in enumerate(genres):
    mask = y_test == i
    if mask.sum() > 0:
        genre_acc = accuracy_score(y_test[mask], y_test_pred[mask])
        print(f"   {genre:15s}: {genre_acc:.2%}")

print("\n" + "=" * 60)
print(f"Gap (Val - Test): {(val_acc - test_acc)*100:.2f}%")
print("=" * 60)