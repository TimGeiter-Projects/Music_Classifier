import pandas as pd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

# ----------------------------
# 1. Pfade & Modell
# ----------------------------
audio_dataset_path = r"/fma_small"
model_path = r"/saved_models/audio_classification.keras"

# Modell laden
model = load_model(model_path)

# ----------------------------
# 2. Metadaten laden
# ----------------------------
tracks = pd.read_csv(
    r"/fma_metadata/tracks.csv",
    index_col=0,
    header=[0, 1]
)

# Titel und Genre auswählen
track_info = tracks.loc[:, [('track','title'), ('track','genre_top')]]
track_info.columns = ['title', 'genre']

# ----------------------------
# 3. Klassenreihenfolge aus deinem Training rekonstruieren
# ----------------------------
# Lade die Features, um die Original-Labels zu bekommen
df = pd.read_pickle("fma_small_features.pkl")
labels = df['class']
class_labels = sorted(labels.unique())  # alphabetische Reihenfolge wie beim pd.get_dummies

# ----------------------------
# 4. Funktion zur Feature-Extraktion
# ----------------------------
def features_extractor(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    return mfcc_scaled_features

# ----------------------------
# 5. Pro Genre 1 Song auswählen & Vorhersage
# ----------------------------
grouped = track_info.groupby('genre')

print("Vorhersagen für 1 Song pro Genre:\n")
for genre, group in grouped:
    # ersten Song nehmen
    track_id = group.index[0]
    title = group.iloc[0]['title']

    # MP3-Pfad konstruieren
    folder = "{:03d}".format(track_id // 1000)
    filename = "{:06d}.mp3".format(track_id)
    file_path = os.path.join(audio_dataset_path, folder, filename)

    # Feature extrahieren
    try:
        feature = features_extractor(file_path)
    except:
        print(f"Fehler bei Track {track_id} ({title})")
        continue

    feature = feature.reshape(1, -1)  # Modell erwartet 2D-Array

    # Vorhersage
    pred_probs = model.predict(feature)
    pred_index = np.argmax(pred_probs)
    pred_class = class_labels[pred_index]

    print(f"Track-ID: {track_id}, Title: {title}, Genre: {genre}")
    print(f"  Modellvorhersage: {pred_class}")
    print("-"*50)
