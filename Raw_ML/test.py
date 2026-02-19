import numpy as np

X      = np.load("X.npy")
y      = np.load("y.npy")
genres = np.load("genres.npy")

print("=== DATEN-CHECK ===")
print(f"X min: {X.min():.4f}, X max: {X.max():.4f}")
print(f"X mean: {X.mean():.4f}, X std: {X.std():.4f}")

print("\n=== LABEL-CHECK ===")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"   Label {u} ({genres[u]}): {c} Tracks")

print("\n=== DUPLIKAT-CHECK ===")
print(f"Sample 0 == Sample 1?   {np.allclose(X[0], X[1])}")
print(f"Sample 0 == Sample 100? {np.allclose(X[0], X[100])}")
print(f"Sample 0 == Sample 500? {np.allclose(X[0], X[500])}")

print("\n=== VARIANZ PRO GENRE ===")
for i, genre in enumerate(genres):
    mask = y == i
    print(f"   {genre}: {X[mask].var():.6f}")