import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ==================== KONFIGURATION ====================
TRACKS_CSV = "../../fma_metadata/tracks.csv"
AUDIO_DIR = "../../fma_small"
OUTPUT_CSV = "3Sekunden.csv"
SEGMENT_LENGTH = 3
TOP_N_FEATURES = 50
TRACKS_PER_CLASS = 200
N_JOBS = max(1, cpu_count() - 1)
TEST_MODE = False
TEST_N_SONGS = 5
RANDOM_SEED = 42

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


def parse_feature_name(feature_name):
    parts = feature_name.split('_')
    if parts[0] == 'spectral':
        feature_type = f"{parts[0]}_{parts[1]}"
        stat_type = parts[2]
        index = int(parts[3])
    elif parts[0] in ['chroma']:
        feature_type = f"{parts[0]}_{parts[1]}"
        stat_type = parts[2]
        index = int(parts[3])
    else:
        feature_type = parts[0]
        stat_type = parts[1]
        index = int(parts[2])
    return feature_type, stat_type, index


def extract_base_features(y, sr):
    features = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc_{i + 1:02d}'] = mfcc[i]
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i in range(12):
        features[f'chroma_cens_{i:02d}'] = chroma_cens[i]
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i in range(12):
        features[f'chroma_cqt_{i:02d}'] = chroma_cqt[i]
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f'chroma_stft_{i:02d}'] = chroma_stft[i]
    rmse = librosa.feature.rms(y=y)
    features['rmse_01'] = rmse[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_01'] = spectral_bandwidth[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_01'] = spectral_centroid[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(7):
        features[f'spectral_contrast_{i + 1:02d}'] = spectral_contrast[i]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_01'] = spectral_rolloff[0]
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for i in range(6):
        features[f'tonnetz_{i + 1:02d}'] = tonnetz[i]
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_01'] = zcr[0]
    return features


def compute_statistics(feature_array):
    return {
        'mean': np.mean(feature_array),
        'std': np.std(feature_array),
        'max': np.max(feature_array),
        'min': np.min(feature_array),
        'median': np.median(feature_array),
        'skew': float(pd.Series(feature_array).skew()),
        'kurtosis': float(pd.Series(feature_array).kurtosis())
    }


def extract_segment_features(y, sr, selected_features):
    base_features = extract_base_features(y, sr)
    segment_features = {}
    for feature_name in selected_features:
        feature_type, stat_type, index = parse_feature_name(feature_name)
        base_key = f"{feature_type}_{index:02d}"
        if base_key in base_features:
            stats = compute_statistics(base_features[base_key])
            segment_features[feature_name] = stats[stat_type]
        else:
            segment_features[feature_name] = 0.0
    return segment_features


def process_single_track(args):
    track_id, genre, audio_dir, segment_length, selected_features = args
    track_id_str = f"{track_id:06d}"
    subdir = track_id_str[:3]
    audio_path = Path(audio_dir) / subdir / f"{track_id_str}.mp3"

    if not audio_path.exists():
        return []

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        segment_samples = segment_length * sr
        segments = []

        for i, start in enumerate(range(0, len(y) - segment_samples, segment_samples)):
            segment = y[start:start + segment_samples]
            features = extract_segment_features(segment, sr, selected_features)
            features['track_id'] = track_id
            features['segment_id'] = i
            features['genre'] = genre
            segments.append(features)

        return segments

    except Exception as e:
        print(f"Fehler bei Track {track_id}: {e}")
        return []


def main():
    print("=" * 60)
    print("Feature-Extraktion für 5-Sekunden-Segmente")
    print("=" * 60)

    # 1. Tracks laden
    print("\n[1/4] Lade Track-Metadaten...")
    tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
    tracks = tracks[tracks[('set', 'subset')] == 'small'].copy()
    tracks = tracks.dropna(subset=[('track', 'genre_top')])

    # NEU: Auf TRACKS_PER_CLASS Tracks pro Genre begrenzen (balanciert)
    tracks = (
        tracks
        .groupby(('track', 'genre_top'), group_keys=False)
        .apply(lambda g: g.sample(min(len(g), TRACKS_PER_CLASS), random_state=RANDOM_SEED))
    )

    track_ids = tracks.index.tolist()
    genres_list = tracks[('track', 'genre_top')].tolist()

    if TEST_MODE:
        track_ids = track_ids[:TEST_N_SONGS]
        genres_list = genres_list[:TEST_N_SONGS]
        print(f"   ⚠ TEST-MODUS: Verarbeite nur {TEST_N_SONGS} Songs")

    print(f"   ✓ {len(track_ids)} Tracks geladen ({TRACKS_PER_CLASS} pro Genre)")
    print(f"   ✓ Verteilung: {tracks[('track', 'genre_top')].value_counts().to_dict()}")
    print(f"   ✓ Segmentlänge: {SEGMENT_LENGTH} Sekunden")

    # 2. Parallel-Verarbeitung
    print(f"\n[2/4] Starte Feature-Extraktion (mit {N_JOBS} Prozessen)...")
    args_list = [
        (track_id, genre, AUDIO_DIR, SEGMENT_LENGTH, SELECTED_FEATURES)
        for track_id, genre in zip(track_ids, genres_list)
    ]

    all_segments = []
    with Pool(processes=N_JOBS) as pool:
        results = list(tqdm(
            pool.imap(process_single_track, args_list),
            total=len(args_list),
            desc="   Verarbeite Tracks"
        ))

    for track_segments in results:
        all_segments.extend(track_segments)

    print(f"\n   ✓ {len(all_segments)} Segmente extrahiert")

    # 3. DataFrame erstellen
    print(f"\n[3/4] Erstelle DataFrame...")
    df = pd.DataFrame(all_segments)
    meta_cols = ['track_id', 'segment_id', 'genre']
    df = df[meta_cols + SELECTED_FEATURES]

    print(f"   ✓ Shape: {df.shape}")
    print(f"   ✓ Segmente pro Genre: {df['genre'].value_counts().to_dict()}")

    # 4. Speichern
    print(f"\n[4/4] Speichere Daten...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"   ✓ Gespeichert als: {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print("Fertig! 🎵")
    print("=" * 60)


if __name__ == "__main__":
    main()