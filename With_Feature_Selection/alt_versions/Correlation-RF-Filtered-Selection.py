import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
import warnings

warnings.filterwarnings('ignore')


TRACKS_CSV = "../fma_metadata/tracks.csv"
FEATURES_CSV = "../fma_metadata/features.csv"
RANDOM_STATE = 42

print("--- Lade Daten ---")
# 1. Daten laden
tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
tracks = tracks[tracks[('set', 'subset')] == 'small'].copy()
tracks = tracks.dropna(subset=[('track', 'genre_top')])

features = pd.read_csv(FEATURES_CSV, index_col=0, header=[0, 1, 2])
features = features.loc[features.index.intersection(tracks.index)]
features = features.loc[tracks.index]

# 2. Labels vorbereiten
y = tracks[('track', 'genre_top')]
label_mapping = y.astype('category').cat.categories
y = y.astype('category').cat.codes
print(f"✓ Labels: {len(label_mapping)} Genres ({', '.join(label_mapping)})")

# 3. Features vorbereiten
X = features.copy()

# NaN Behandlung
if X.isna().sum().sum() > 0:
    valid_indices = ~X.isna().any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]

# Spaltennamen flachklopfen
X.columns = ['_'.join(map(str, col)).strip() for col in X.columns.values]
print(f"✓ Rohdaten Dimension: {X.shape}")

# Korrelations-Analyse & Bereinigung (VOR dem Split)
print("\n--- A. Bereinigung (Korrelation) ---")
corr_matrix = X.corr().abs()

# Obere Dreiecksmatrix (um Dopplungen zu vermeiden)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Finde Spalten mit > 0.95 Korrelation
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(f"⚠ Entferne {len(to_drop)} redundante Features (Corr > 0.95)")

X = X.drop(columns=to_drop)
print(f"✓ Dimension nach Bereinigung: {X.shape}")


#Split & Scaling
print("\n--- B. Split & Scaling ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Daten skaliert (StandardScaler)")


# Feature Selection (Random Forest)
print("\n--- C. Feature Selection (RF) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Selektion basierend auf Wichtigkeit
selector = SelectFromModel(rf, prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"✓ Features reduziert von {X_train_scaled.shape[1]} auf {X_train_selected.shape[1]}")

# Top Features anzeigen
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
current_cols = X.columns
print("Top 5 wichtigste Features:")
for f in range(5):
    # Mapping auf die ursprünglichen Spaltenindizes
    print(f"  {f+1}. {current_cols[indices[f]]} ({importances[indices[f]]:.4f})")

# OPTIONAL: Heatmap der Top 30 korrelierten Features
print("\n--- Erstelle Korrelation-Heatmap (Top 30 Features) ---")
# Wähle die 30 wichtigsten Features
top_n = 30
top_features = current_cols[indices[:top_n]]
corr_top = X[top_features].corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr_top, cmap="coolwarm", annot=False)
plt.title("Korrelations-Heatmap – Top 30 Features")
plt.tight_layout()
plt.savefig("heatmap_top30_features.png", dpi=300)
plt.show()
print("✓ Heatmap gespeichert als: heatmap_top30_features.png")

# SVM Training (GridSearch)
print("\n--- D. SVM GridSearch Training ---")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001], # gamma=1 sehr aggressiv
    'kernel': ['rbf']
}

# n_jobs=-1 nutzt alle CPU Kerne
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=3, n_jobs=-1)
grid.fit(X_train_selected, y_train)

print(f"✓ Beste Parameter: {grid.best_params_}")


# Auswertung
print("\n--- E. Ergebnis ---")
y_pred = grid.predict(X_test_selected)

acc = accuracy_score(y_test, y_pred)
print(f"FINAL ACCURACY: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping))

# Confusion Matrix Visualisierung
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_mapping, yticklabels=label_mapping)
plt.ylabel('Wahre Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.title(f'Confusion Matrix (Acc: {acc:.2%})')
plt.tight_layout()
plt.show()