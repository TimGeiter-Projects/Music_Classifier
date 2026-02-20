import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ==================== LADEN ====================
X_test        = np.load("X_test_v1.npy")
y_test        = np.load("y_test_v1.npy")
label_mapping = np.load("label_mapping_v1.npy", allow_pickle=True)

models = {
    'SVM':               joblib.load("svm_model_v1.pkl"),
    'Random Forest':     joblib.load("rf_model_v1.pkl"),
    'k-NN':              joblib.load("knn_model_v1.pkl"),
    'Gradient Boosting': joblib.load("gb_model_v1.pkl"),
}

# ==================== EVALUATION ALLE MODELLE ====================
results_summary = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Modell: {name}")
    print(f"{'='*50}")

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    results_summary[name] = acc

    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=label_mapping))

# ==================== CONFUSION MATRICES ====================
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()
cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']

for ax, (name, model), cmap in zip(axes, models.items(), cmaps):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = results_summary[name]
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=label_mapping, yticklabels=label_mapping, ax=ax)
    ax.set_title(f'{name}\nAcc: {acc:.2%}')
    ax.set_xlabel('Vorhergesagt')
    ax.set_ylabel('Wahre Klasse')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("eval_v1_confusion_matrices.png", dpi=300)
plt.show()

# ==================== ZUSAMMENFASSUNG ====================
print(f"\n{'='*50}")
print("ZUSAMMENFASSUNG V1 (StandardScaler + RF Selection)")
print(f"{'='*50}")
for name, acc in sorted(results_summary.items(), key=lambda x: x[1], reverse=True):
    print(f"   {name:20s}: {acc:.2%}")
print(f"{'='*50}")