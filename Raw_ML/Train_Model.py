#Data Preprocessing
import os
import numpy as np
import pandas as pd
import librosa
from skimage.transform import resize
from tqdm import tqdm

# ----------------------------
# Konfiguration
# ----------------------------
AUDIO_DATASET_PATH = "../fma_small"
METADATA_PATH      = "../fma_metadata/tracks.csv"
OUTPUT_DIR         = "."

SR_TARGET          = 22050
N_MELS             = 128
SHAPE              = (128, 128)
TRACKS_PER_CLASS   = 1000
SEGMENT_DURATION   = 5      # Sekunden pro Segment
N_SEGMENTS         = 6      # 6 × 5s = 30s pro Track
RANDOM_SEED        = 42

np.random.seed(RANDOM_SEED)

# ----------------------------
# 1. Metadaten laden & filtern
# ----------------------------
print("=" * 60)
print("Preprocessing: Mel-Spektrogramme aus 5s-Segmenten")
print("=" * 60)

tracks = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])

fma_small_tracks = tracks[tracks[('set', 'subset')] == 'small'].copy()
fma_small_tracks = fma_small_tracks.dropna(subset=[('track', 'genre_top')])

small_basic = pd.DataFrame({
    'title':     fma_small_tracks[('track', 'title')].values,
    'genre_top': fma_small_tracks[('track', 'genre_top')].values
}, index=fma_small_tracks.index)
small_basic.index.name = "track_id"

# Auf TRACKS_PER_CLASS Tracks pro Klasse begrenzen (balanciert)
small_basic = (
    small_basic
    .groupby('genre_top', group_keys=False)
    .apply(lambda g: g.sample(min(len(g), TRACKS_PER_CLASS), random_state=RANDOM_SEED))
)

print(f"\nTracks gesamt: {len(small_basic)}")
print("Verteilung:\n", small_basic['genre_top'].value_counts())

genres       = sorted(small_basic['genre_top'].unique())
genre_to_idx = {g: i for i, g in enumerate(genres)}
print("\nGenre Mapping:", genre_to_idx)

# ----------------------------
# 2. Hilfsfunktionen
# ----------------------------
def get_fma_small_path(base_dir, track_id):
    folder   = "{:03d}".format(track_id // 1000)
    filename = "{:06d}.mp3".format(track_id)
    return os.path.join(base_dir, folder, filename)


def segment_to_melspectrogram(segment, sr):
    """Einen 5s Audio-Ausschnitt in ein normalisiertes Mel-Spektrogramm umwandeln."""
    mel        = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS)
    mel_db     = librosa.power_to_db(mel, ref=np.max)
    mel_resize = resize(mel_db, SHAPE, mode='reflect', anti_aliasing=True).astype(np.float32)

    mel_min, mel_max = mel_resize.min(), mel_resize.max()
    if mel_max - mel_min > 0:
        mel_resize = (mel_resize - mel_min) / (mel_max - mel_min)
    else:
        mel_resize = np.zeros_like(mel_resize)

    return np.expand_dims(mel_resize, axis=-1)  # (128, 128, 1)


# ----------------------------
# 3. Dataset erstellen
# WICHTIG: Track-ID wird gespeichert für korrekten Split später
# ----------------------------
print("\n[1/2] Erstelle Segmente aus Tracks...")

X          = []
y          = []
track_ids  = []   # NEU: Track-ID pro Segment merken für korrekten Split
skipped    = 0
segment_samples = SEGMENT_DURATION * SR_TARGET  # 5 * 22050 = 110250

for track_id, row in tqdm(small_basic.iterrows(), total=len(small_basic)):
    file_path = get_fma_small_path(AUDIO_DATASET_PATH, track_id)
    if not os.path.exists(file_path):
        skipped += 1
        continue

    try:
        audio, sr = librosa.load(file_path, sr=SR_TARGET, mono=True)

        # Track auf genau 30s bringen
        target_samples = N_SEGMENTS * segment_samples
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        else:
            audio = audio[:target_samples]

        # 6 Segmente à 5s extrahieren
        label = genre_to_idx[row['genre_top']]
        for i in range(N_SEGMENTS):
            start   = i * segment_samples
            segment = audio[start:start + segment_samples]
            mel     = segment_to_melspectrogram(segment, sr)

            X.append(mel)
            y.append(label)
            track_ids.append(track_id)   # Alle Segmente eines Tracks haben dieselbe ID

    except Exception as e:
        print(f"Fehler bei Track {track_id}: {e}")
        skipped += 1

X         = np.array(X, dtype=np.float32)
y         = np.array(y, dtype=np.int32)
track_ids = np.array(track_ids, dtype=np.int32)

print(f"\n{len(X)} Segmente aus {len(small_basic) - skipped} Tracks erstellt.")
print(f"{skipped} Tracks übersprungen.")
print(f"X Shape: {X.shape}")

# ----------------------------
# 4. Track-Level Split
# WICHTIG: Alle Segmente eines Tracks müssen in dieselbe Partition!
# Sonst würde das Modell beim Test "schummeln" (Data Leakage)
# ----------------------------
print("\n[2/2] Track-Level Split (90% Train+Val / 10% Test)...")

unique_tracks = np.unique(track_ids)
np.random.shuffle(unique_tracks)

n_test_tracks  = int(len(unique_tracks) * 0.10)
test_track_ids = set(unique_tracks[:n_test_tracks])
train_track_ids = set(unique_tracks[n_test_tracks:])

train_mask = np.array([tid in train_track_ids for tid in track_ids])
test_mask  = np.array([tid in test_track_ids  for tid in track_ids])

X_trainval = X[train_mask]
y_trainval = y[train_mask]
X_test     = X[test_mask]
y_test     = y[test_mask]

print(f"   ✓ Train+Val Segmente: {len(X_trainval)}  (aus {len(train_track_ids)} Tracks)")
print(f"   ✓ Test Segmente:      {len(X_test)}  (aus {len(test_track_ids)} Tracks)")

# Klassenverteilung checken
print("\n   Klassenverteilung Test:")
for i, genre in enumerate(genres):
    count = (y_test == i).sum()
    print(f"      {genre}: {count} Segmente")

# ----------------------------
# 5. Speichern
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, "X.npy"),          X_trainval)
np.save(os.path.join(OUTPUT_DIR, "y.npy"),          y_trainval)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),     X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),     y_test)
np.save(os.path.join(OUTPUT_DIR, "genres.npy"),     np.array(genres))

print("\n✓ Gespeichert:")
print(f"   X.npy      → {X_trainval.shape}  (Train+Val)")
print(f"   y.npy      → {y_trainval.shape}")
print(f"   X_test.npy → {X_test.shape}  (Test, nicht anfassen!)")
print(f"   y_test.npy → {y_test.shape}")
print(f"   genres.npy → {genres}")
print("\nFertig!")