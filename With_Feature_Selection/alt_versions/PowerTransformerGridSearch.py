import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer # Der Gamechanger
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 1. Daten laden (Identisch mit deinem Original)
TRACKS_CSV = "../fma_metadata/tracks.csv"
FEATURES_CSV = "../fma_metadata/features.csv"

print("--- Lade Daten ---")
tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
tracks = tracks[tracks[('set', 'subset')] == 'small'].copy()
tracks = tracks.dropna(subset=[('track', 'genre_top')])

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

# 2. Dein Korrelations-Filter (0.95) - Wir behalten ihn, er ist gut!
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. PowerTransformer statt StandardScaler
# Das hilft der SVM massiv, Pop von Experimental zu trennen,
# da die Verteilungen dieser Genres oft sehr "schief" sind.
print("\n--- Transformiere Daten (Yeo-Johnson) ---")
scaler = PowerTransformer(method='yeo-johnson')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. GridSearch mit Fokus auf höheres C und Class Weights
# 'balanced' sorgt dafür, dass die SVM Pop nicht einfach bevorzugt, weil es "sicherer" ist.
print("--- Starte GridSearch ---")
param_grid = {
    'C': [10, 20, 50],           # Härtere Bestrafung für Fehlklassifizierung
    'gamma': ['scale', 0.01],    # scale nutzt 1 / (n_features * X.var())
    'kernel': ['rbf'],
    'class_weight': ['balanced'] # Hilft gegen die Pop/Experimental-Verwechslung
}

grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

# 6. Auswertung
y_pred = grid.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\nNEUER REKORDVERSUCH: {acc:.2%}")
print(f"Beste Parameter: {grid.best_params_}")

# Confusion Matrix zur Kontrolle der Pop-Verwechslung
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=label_mapping, yticklabels=label_mapping)
plt.title(f'Verbesserte Matrix (Acc: {acc:.2%})')
plt.show()