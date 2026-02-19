import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==================== KONFIGURATION ====================
SEGMENTS_CSV = "dreiSekunden.csv"
MODEL_OUTPUT = "svm_segment_model.pkl"
SCALER_OUTPUT = "scaler_segment_model.pkl"
RANDOM_STATE = 42

# Top 50 Features (die wir berechnet haben)
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

print("=" * 60)
print("SVM Training mit 3-Sekunden-Segment Features")
print("=" * 60)

# 1. Daten laden
print("\n[1/5] Lade Segment-Daten...")
df = pd.read_csv(SEGMENTS_CSV)

print(f"   ✓ Shape: {df.shape}")
print(f"   ✓ Segmente pro Genre:")
for genre, count in df['genre'].value_counts().items():
    print(f"      - {genre}: {count}")

# 2. Features und Labels vorbereiten
print("\n[2/5] Bereite Features und Labels vor...")
X = df[SELECTED_FEATURES]
y = df['genre']

# Labels in numerische Codes umwandeln
label_mapping = y.astype('category').cat.categories
y = y.astype('category').cat.codes

print(f"   ✓ Features: {X.shape}")
print(f"   ✓ Labels: {len(label_mapping)} Genres ({', '.join(label_mapping)})")

# NaN Check
if X.isna().sum().sum() > 0:
    print(f"   ⚠ {X.isna().sum().sum()} NaN-Werte gefunden - werden mit 0 gefüllt")
    X = X.fillna(0)

# 3. Split & Scaling
print("\n[3/5] Split & Scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✓ Training: {X_train_scaled.shape[0]} Segmente")
print(f"   ✓ Test: {X_test_scaled.shape[0]} Segmente")
print("   ✓ Daten skaliert (StandardScaler)")

# 4. SVM Training (GridSearch)
print("\n[4/5] SVM GridSearch Training...")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=3, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print(f"\n   ✓ Beste Parameter: {grid.best_params_}")

# 5. Auswertung
print("\n[5/5] Evaluierung...")
y_pred = grid.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"FINAL ACCURACY (3-Sekunden-Segmente): {acc:.2%}")
print(f"{'='*60}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping))

# Confusion Matrix Visualisierung
print("\n--- Erstelle Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_mapping, yticklabels=label_mapping)
plt.ylabel('Wahre Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.title(f'Confusion Matrix - 3-Sekunden-Segmente (Acc: {acc:.2%})')
plt.tight_layout()
plt.savefig("confusion_matrix_segments.png", dpi=300)
plt.show()
print("   ✓ Confusion Matrix gespeichert als: confusion_matrix_segments.png")

# Accuracy pro Genre
print("\n--- Accuracy pro Genre ---")
for i, genre in enumerate(label_mapping):
    genre_mask = y_test == i
    if genre_mask.sum() > 0:
        genre_acc = accuracy_score(y_test[genre_mask], y_pred[genre_mask])
        print(f"   {genre}: {genre_acc:.2%}")

# Feature Importance Visualisierung (basierend auf Support Vectors)
print("\n--- Top 10 wichtigste Features (nach SVM) ---")
# Für lineare SVM könnten wir coef_ nutzen, für RBF ist das schwieriger
# Stattdessen: Feature-Permutation oder einfach die Top-Features ausgeben
for i, feat in enumerate(SELECTED_FEATURES[:10], 1):
    print(f"   {i}. {feat}")

# Modell und Scaler speichern
print("\n--- Speichere Modell ---")
joblib.dump(grid.best_estimator_, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)
print(f"   ✓ Modell gespeichert als: {MODEL_OUTPUT}")
print(f"   ✓ Scaler gespeichert als: {SCALER_OUTPUT}")

print("\n" + "=" * 60)
print("Training abgeschlossen! 🎵")
print("=" * 60)

# Vergleich mit Song-Level (falls relevant)
print("\nHinweis:")
print("  - Dieses Modell klassifiziert 3-Sekunden-Segmente")
print("  - Für Song-Level Prediction: Mehrheitsvotum über alle Segmente")
print(f"  - Modell laden: model = joblib.load('{MODEL_OUTPUT}')")