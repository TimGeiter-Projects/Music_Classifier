import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import warnings

warnings.filterwarnings('ignore')

TRACKS_CSV = "../../fma_metadata/tracks.csv"
FEATURES_CSV = "../../fma_metadata/features.csv"
RANDOM_STATE = 42
TRACKS_PER_CLASS = 200

print("--- Lade Daten ---")
tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
tracks = tracks[tracks[('set', 'subset')] == 'small'].copy()
tracks = tracks.dropna(subset=[('track', 'genre_top')])

tracks = (
    tracks
    .groupby(('track', 'genre_top'), group_keys=False)
    .apply(lambda g: g.sample(min(len(g), TRACKS_PER_CLASS), random_state=RANDOM_STATE))
)
print(f"✓ Tracks nach Balancing: {len(tracks)} ({TRACKS_PER_CLASS} pro Genre)")

features = pd.read_csv(FEATURES_CSV, index_col=0, header=[0, 1, 2])
features = features.loc[features.index.intersection(tracks.index)]
features = features.loc[tracks.index]

y = tracks[('track', 'genre_top')]
label_mapping = y.astype('category').cat.categories
y = y.astype('category').cat.codes

X = features.copy()
X.columns = ['_'.join(map(str, col)).strip() for col in X.columns.values]
if X.isna().sum().sum() > 0:
    X = X.fillna(X.median())

# Korrelations-Bereinigung
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop)
print(f"✓ Dimension nach Korrelations-Bereinigung: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# PowerTransformer_Balanced
print("\n--- Transformiere Daten (Yeo-Johnson) ---")
scaler = PowerTransformer(method='yeo-johnson')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==================== MODELL 1: SVM ====================
print("\n--- D1. SVM GridSearch ---")
svm_grid = GridSearchCV(
    SVC(probability=True),
    {'C': [10, 20, 50], 'gamma': ['scale', 0.01], 'kernel': ['rbf'], 'class_weight': ['balanced']},
    cv=3, n_jobs=-1, verbose=1
)
svm_grid.fit(X_train_scaled, y_train)
print(f"✓ SVM beste Parameter: {svm_grid.best_params_}")

# ==================== MODELL 2: Random Forest ====================
print("\n--- D2. Random Forest GridSearch ---")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    {'n_estimators': [100, 200], 'max_depth': [None, 20, 40], 'min_samples_split': [2, 5]},
    cv=3, n_jobs=-1, verbose=1
)
rf_grid.fit(X_train_scaled, y_train)
print(f"✓ RF beste Parameter: {rf_grid.best_params_}")

# ==================== MODELL 3: k-NN ====================
print("\n--- D3. k-NN GridSearch ---")
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
    cv=3, n_jobs=-1, verbose=1
)
knn_grid.fit(X_train_scaled, y_train)
print(f"✓ k-NN beste Parameter: {knn_grid.best_params_}")

# ==================== MODELL 4: Gradient Boosting ====================
print("\n--- D4. Gradient Boosting GridSearch ---")
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    cv=3, n_jobs=-1, verbose=1
)
gb_grid.fit(X_train_scaled, y_train)
print(f"✓ GB beste Parameter: {gb_grid.best_params_}")

# ==================== LEARNING CURVES ====================
print("\n--- E. Learning Curves ---")
models = {
    'SVM':               svm_grid.best_estimator_,
    'Random Forest':     rf_grid.best_estimator_,
    'k-NN':              knn_grid.best_estimator_,
    'Gradient Boosting': gb_grid.best_estimator_,
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=3, n_jobs=-1
    )
    ax.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score")
    ax.plot(train_sizes, np.mean(val_scores,   axis=1), label="Validation Score")
    ax.set_title(f"Learning Curve – {name}")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.savefig("learning_curves_all_models_v2.png", dpi=300)
plt.show()
print("✓ Learning Curves gespeichert")

# ==================== SPEICHERN ====================
joblib.dump(svm_grid.best_estimator_, "svm_model_v2.pkl")
joblib.dump(rf_grid.best_estimator_,  "rf_model_v2.pkl")
joblib.dump(knn_grid.best_estimator_, "knn_model_v2.pkl")
joblib.dump(gb_grid.best_estimator_,  "gb_model_v2.pkl")
joblib.dump(scaler,                   "scaler_v2.pkl")

np.save("X_test_v2.npy",        X_test_scaled)
np.save("y_test_v2.npy",        y_test.values)
np.save("label_mapping_v2.npy", np.array(label_mapping))

print("\n✓ Gespeichert: svm_model_v2.pkl, rf_model_v2.pkl, knn_model_v2.pkl, gb_model_v2.pkl")
print("✓ Gespeichert: scaler_v2.pkl")
print("✓ Gespeichert: X_test_v2.npy, y_test_v2.npy, label_mapping_v2.npy")