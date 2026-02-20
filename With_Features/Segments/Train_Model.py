from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

# ==================== KONFIGURATION ====================
SEGMENTS_CSV = "3Sekunden.csv"
MODEL_OUTPUT = "svm_segment_modelv2.pkl"
SCALER_OUTPUT = "scaler_segment_modelv2.pkl"
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

# ==================== DATEN LADEN ====================
df = pd.read_csv(SEGMENTS_CSV)

y = df['genre'].astype('category').cat.codes
label_mapping = df['genre'].astype('category').cat.categories

X = df[SELECTED_FEATURES]
groups = df['track_id'].values

# ==================== GROUP SPLIT (90/10 Songs) ====================
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

# ==================== SCALING ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ==================== GRID SEARCH MIT GroupKFold ====================
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

cv = GroupKFold(n_splits=3)

grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_scaled, y_train, groups=groups[train_idx])

print("Beste Parameter:", grid.best_params_)

# ==================== LEARNING CURVE ====================
train_sizes, train_scores, val_scores = learning_curve(
    grid.best_estimator_,
    X_train_scaled,
    y_train,
    groups=groups[train_idx],
    cv=cv,
    train_sizes=np.linspace(0.3, 1.0, 5),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, val_mean, label="Validation Score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (GroupKFold)")
plt.legend()
plt.grid()
plt.savefig("Learning_Curve_V2.png", dpi=300)
plt.show()
print("   ✓ Learning Curve gespeichert")

# ==================== SPEICHERN ====================
joblib.dump(grid.best_estimator_, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)

print(f"\nModell gespeichert: {MODEL_OUTPUT}")
print(f"Scaler gespeichert: {SCALER_OUTPUT}")