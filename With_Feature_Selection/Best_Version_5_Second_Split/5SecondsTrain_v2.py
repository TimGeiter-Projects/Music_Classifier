from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==================== KONFIGURATION ====================
SEGMENTS_CSV = "dreiSekunden.csv"
MODEL_OUTPUT = "svm_segment_model.pkl"
SCALER_OUTPUT = "scaler_segment_model.pkl"
RANDOM_STATE = 42
SELECTED_FEATURES = [  # Deine Top 50 Features
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

# ------------------------ Daten laden ------------------------
df = pd.read_csv(SEGMENTS_CSV)

# Labels
y = df['genre'].astype('category').cat.codes
label_mapping = df['genre'].astype('category').cat.categories

# Features
X = df[SELECTED_FEATURES]

# Gruppen: alle Segmente eines Songs gehören zusammen
groups = df['track_id'].values

# ------------------------ 90/10 Split nach track_id ------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ------------------------ Scaling ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------ GridSearchCV ------------------------
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

# ------------------------ Evaluation ------------------------
y_pred = grid.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Final Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, target_names=label_mapping))

# ------------------------ Modell & Scaler speichern ------------------------
joblib.dump(grid.best_estimator_, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)
print(f"Modell gespeichert: {MODEL_OUTPUT}")
print(f"Scaler gespeichert: {SCALER_OUTPUT}")
