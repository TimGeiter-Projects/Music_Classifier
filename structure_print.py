import pandas as pd
import os

# -----------------------------
# Pfad zu deinem fma_metadata Ordner
# -----------------------------
metadata_path = r"C:\Users\geite\Desktop\Uni\Master\WS25_26\Maschinelles_Lernen\music_classifier\fma_metadata"



def print_section(title):
    print("\n" + "="*80)
    print(" " + title)
    print("="*80)


# -----------------------------
# 1. tracks.csv
# -----------------------------
print_section("tracks.csv laden")

tracks_file = os.path.join(metadata_path, "tracks.csv")

tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])

print("\n▶ tracks.csv — Struktur")
print(tracks.info())

print("\n▶ tracks.csv — Spalten-Level-0:")
print(tracks.columns.levels[0])

print("\n▶ tracks.csv — Spalten-Level-1 Beispiel:")
for group in tracks.columns.levels[0]:
    print(f"  {group}: {list(tracks[group].columns[:10])}")

print("\n▶ Beispiel (erste 3 Zeilen):")
print(tracks.head(3))


# -----------------------------
# 2. genres.csv
# -----------------------------
print_section("genres.csv laden")

genres_file = os.path.join(metadata_path, "genres.csv")

genres = pd.read_csv(genres_file, index_col=0)

print("\n▶ genres.csv — Struktur")
print(genres.info())

print("\n▶ Spalten:")
print(genres.columns)

print("\n▶ Beispiel (erste 5 Zeilen):")
print(genres.head())


# -----------------------------
# 3. echonest.csv (falls vorhanden)
# -----------------------------
echonest_file = os.path.join(metadata_path, "echonest.csv")

if os.path.exists(echonest_file):
    print_section("echonest.csv laden")

    echonest = pd.read_csv(echonest_file, index_col=0, header=[0, 1])

    print("\n▶ echonest.csv — Struktur")
    print(echonest.info())

    print("\n▶ Spalten-Level-0:")
    print(echonest.columns.levels[0])

    print("\n▶ Beispiel (erste 5 Zeilen):")
    print(echonest.head())

else:
    print_section("echonest.csv NICHT gefunden – wahrscheinlich nicht in fma_small")


print("\n\nFERTIG! 🎉")
