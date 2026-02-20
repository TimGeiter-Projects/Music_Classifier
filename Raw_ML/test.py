import pandas as pd

# ----------------------------
# Metadaten laden
# ----------------------------
METADATA_PATH = "../fma_metadata/tracks.csv"

print("=" * 60)
print("FMA Small - Songs pro Genre")
print("=" * 60)

# CSV laden (Header hat 2 Zeilen)
tracks = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])

# Nur FMA Small subset filtern
fma_small = tracks[tracks[('set', 'subset')] == 'small'].copy()
fma_small = fma_small.dropna(subset=[('track', 'genre_top')])

# Genre-Spalte extrahieren
genres = fma_small[('track', 'genre_top')]

print(f"\nGesamtanzahl Tracks in FMA Small: {len(fma_small)}")
print("\n--- Songs pro Genre ---")

# Zählen und sortieren
genre_counts = genres.value_counts().sort_index()

for genre, count in genre_counts.items():
    print(f"   {genre:20s}: {count:4d} Tracks")

print("\n" + "=" * 60)
print(f"Genres gesamt: {len(genre_counts)}")
print("=" * 60)