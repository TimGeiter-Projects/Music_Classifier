import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

# ==================== KONFIGURATION ====================
DATA_DIR   = r"C:\Users\geite\Desktop\Uni\Master\WS25_26\Maschinelles_Lernen\music_classifier\Raw_ML"
MODEL_PATH = os.path.join(DATA_DIR, "best_model.keras")

# ==================== LADEN ====================
model          = tf.keras.models.load_model(MODEL_PATH)
X_test         = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test         = np.load(os.path.join(DATA_DIR, "y_test.npy"))
track_ids_test = np.load(os.path.join(DATA_DIR, "track_ids_test.npy"))
genres         = np.load(os.path.join(DATA_DIR, "genres.npy"))

print(f"✓ Test: {X_test.shape[0]} Segmente ({len(np.unique(track_ids_test))} Songs)")

# ==================== SEGMENT-LEVEL ====================
probs  = model.predict(X_test)
y_pred = np.argmax(probs, axis=1)

seg_acc = accuracy_score(y_test, y_pred)
print(f"\nSegment-Level Accuracy: {seg_acc:.2%}")
print(classification_report(y_test, y_pred, target_names=genres))

# ==================== MAJORITY VOTING ====================
results = pd.DataFrame({
    'track_id': track_ids_test,
    'true':     y_test,
    'pred':     y_pred
})

song_majority = results.groupby('track_id').agg(
    true=('true', 'first'),
    pred=('pred', lambda x: x.mode()[0])
).reset_index()

majority_acc = accuracy_score(song_majority['true'], song_majority['pred'])
print(f"\nSong-Level Accuracy (Majority Voting): {majority_acc:.2%}")
print(classification_report(song_majority['true'], song_majority['pred'], target_names=genres))

# ==================== WEIGHTED VOTING ====================
genre_cols = list(range(len(genres)))

probs_df = pd.DataFrame(probs, columns=genre_cols)
probs_df['track_id'] = track_ids_test
probs_df['true']     = y_test

song_probs    = probs_df.groupby('track_id')[genre_cols].mean()  # ← genre_cols statt range()
song_true     = probs_df.groupby('track_id')['true'].first()
weighted_pred = song_probs.values.argmax(axis=1)


weighted_acc  = accuracy_score(song_true, weighted_pred)
print(f"\nSong-Level Accuracy (Weighted Voting): {weighted_acc:.2%}")
print(classification_report(song_true, weighted_pred, target_names=genres))

# ==================== CONFUSION MATRICES ====================
fig, axes = plt.subplots(1, 3, figsize=(28, 8))

# Segment-Level
cm_seg = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_seg, annot=True, fmt='d', cmap='Blues',
            xticklabels=genres, yticklabels=genres, ax=axes[0])
axes[0].set_title(f"Segment-Ebene\nGenauigkeit: {seg_acc:.2%}")
axes[0].set_xlabel("Vorhergesagte Klasse")
axes[0].set_ylabel("Wahre Klasse")
axes[0].tick_params(axis='x', rotation=45)

# Majority Voting
cm_maj = confusion_matrix(song_majority['true'], song_majority['pred'])
sns.heatmap(cm_maj, annot=True, fmt='d', cmap='Greens',
            xticklabels=genres, yticklabels=genres, ax=axes[1])
axes[1].set_title(f"Song-Ebene – Mehrheitswahl\nGenauigkeit: {majority_acc:.2%}")
axes[1].set_xlabel("Vorhergesagte Klasse")
axes[1].set_ylabel("Wahre Klasse")
axes[1].tick_params(axis='x', rotation=45)

# Weighted Voting
cm_wei = confusion_matrix(song_true, weighted_pred)
sns.heatmap(cm_wei, annot=True, fmt='d', cmap='Oranges',
            xticklabels=genres, yticklabels=genres, ax=axes[2])
axes[2].set_title(f"Song-Ebene – Gewichtete Wahl\nGenauigkeit: {weighted_acc:.2%}")
axes[2].set_xlabel("Vorhergesagte Klasse")
axes[2].set_ylabel("Wahre Klasse")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.9)
plt.savefig(os.path.join(DATA_DIR, "eval_cnn_confusion_matrices.png"), dpi=300)
plt.show()