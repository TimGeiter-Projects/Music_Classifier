from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

# ==================== KONFIGURATION ====================
SEGMENTS_CSV = "3Sekunden.csv"
MODEL_PATH   = "svm_segment_modelv2.pkl"
SCALER_PATH  = "scaler_segment_modelv2.pkl"
RANDOM_STATE = 42

SELECTED_FEATURES = [
    'spectral_contrast_mean_04', 'mfcc_mean_03', 'mfcc_mean_01', 'mfcc_max_04',
    'spectral_bandwidth_mean_01', 'spectral_centroid_mean_01', 'spectral_contrast_mean_03',
    'spectral_contrast_mean_02', 'zcr_mean_01', 'spectral_contrast_mean_07',
    'spectral_rolloff_skew_01', 'mfcc_max_01', 'spectral_contrast_std_04',
    'rmse_std_01', 'zcr_median_01', 'spectral_rolloff_kurtosis_01',
    'mfcc_std_02', 'spectral_centroid_std_01', 'spectral_contrast_skew_02',
    'mfcc_max_02', 'mfcc_std_04', 'tonnetz_std_01', 'mfcc_std_14',
    'mfcc_std_13', 'mfcc_mean_18', 'tonnetz_std_05', 'spectral_contrast_skew_03',
    'mfcc_mean_02', 'spectral_contrast_mean_05', 'spectral_centroid_kurtosis_01',
    'mfcc_std_17', 'spectral_bandwidth_skew_01', 'mfcc_kurtosis_04',
    'mfcc_min_05', 'tonnetz_std_02', 'mfcc_max_06', 'mfcc_std_08',
    'mfcc_skew_01', 'mfcc_mean_16', 'mfcc_std_20', 'mfcc_skew_04',
    'mfcc_std_06', 'tonnetz_std_04', 'spectral_centroid_skew_01',
    'mfcc_mean_20', 'spectral_bandwidth_std_01', 'spectral_contrast_max_07',
    'mfcc_std_19', 'mfcc_std_16', 'tonnetz_std_06'
]

# ==================== DATEN & MODELL LADEN ====================
df     = pd.read_csv(SEGMENTS_CSV)
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

y             = df['genre'].astype('category').cat.codes
label_mapping = df['genre'].astype('category').cat.categories
X             = df[SELECTED_FEATURES]
groups        = df['track_id'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
_, test_idx = next(gss.split(X, y, groups=groups))

X_test         = X.iloc[test_idx]
y_test         = y.iloc[test_idx]
track_ids_test = groups[test_idx]
X_test_scaled  = scaler.transform(X_test)

# ==================== SEGMENT-LEVEL ====================
y_pred  = model.predict(X_test_scaled)
seg_acc = accuracy_score(y_test, y_pred)
print(f"\nSegment-Level Accuracy: {seg_acc:.2%}")
print(classification_report(y_test, y_pred, target_names=label_mapping))

# ==================== MAJORITY VOTING ====================
results = pd.DataFrame({
    'track_id': track_ids_test,
    'true':     y_test.values,
    'pred':     y_pred
})

song_majority = results.groupby('track_id').agg(
    true=('true', 'first'),
    pred=('pred', lambda x: x.mode()[0])
).reset_index()

majority_acc = accuracy_score(song_majority['true'], song_majority['pred'])
print(f"\nSong-Level Accuracy (Majority Voting): {majority_acc:.2%}")
print(classification_report(song_majority['true'], song_majority['pred'], target_names=label_mapping))

# ==================== WEIGHTED VOTING ====================
probs      = model.predict_proba(X_test_scaled)
genre_cols = list(range(len(label_mapping)))

probs_df             = pd.DataFrame(probs, columns=genre_cols)
probs_df['track_id'] = track_ids_test
probs_df['true']     = y_test.values

song_probs    = probs_df.groupby('track_id')[genre_cols].mean()
song_true     = probs_df.groupby('track_id')['true'].first()
weighted_pred = song_probs.values.argmax(axis=1)

weighted_acc = accuracy_score(song_true, weighted_pred)
print(f"\nSong-Level Accuracy (Weighted Voting): {weighted_acc:.2%}")
print(classification_report(song_true, weighted_pred, target_names=label_mapping))

# ==================== CONFUSION MATRICES ====================
fig, axes = plt.subplots(1, 3, figsize=(28, 8))

# Segment-Level
cm_seg = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm_seg, display_labels=label_mapping).plot(
    ax=axes[0], cmap="Blues", xticks_rotation=45)
axes[0].set_title(f"Segment-Ebene\nGenauigkeit: {seg_acc:.2%}")
axes[0].set_xlabel("Vorhergesagte Klasse")
axes[0].set_ylabel("Wahre Klasse")

# Majority Voting
cm_maj = confusion_matrix(song_majority['true'], song_majority['pred'])
ConfusionMatrixDisplay(cm_maj, display_labels=label_mapping).plot(
    ax=axes[1], cmap="Greens", xticks_rotation=45)
axes[1].set_title(f"Song-Ebene – Mehrheitswahl\nGenauigkeit: {majority_acc:.2%}")
axes[1].set_xlabel("Vorhergesagte Klasse")
axes[1].set_ylabel("Wahre Klasse")

# Weighted Voting
cm_wei = confusion_matrix(song_true, weighted_pred)
ConfusionMatrixDisplay(cm_wei, display_labels=label_mapping).plot(
    ax=axes[2], cmap="Oranges", xticks_rotation=45)
axes[2].set_title(f"Song-Ebene – Gewichtete Wahl\nGenauigkeit: {weighted_acc:.2%}")
axes[2].set_xlabel("Vorhergesagte Klasse")
axes[2].set_ylabel("Wahre Klasse")

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.9)
plt.savefig("eval_svm_confusion_matrices.png", dpi=300)
plt.show()

print(f"\n{'='*50}")
print(f"Segment-Level Acc  : {seg_acc:.2%}")
print(f"Majority Voting Acc: {majority_acc:.2%}")
print(f"Weighted Voting Acc: {weighted_acc:.2%}")
print(f"{'='*50}")