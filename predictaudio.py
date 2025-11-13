import numpy as np
import os, librosa, utils
from predictor import predict, load_model

def extract_mfcc(y, sr, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_var  = np.var(mfccs, axis=1)
    return np.concatenate([mfcc_mean, mfcc_var])

def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_var  = np.var(chroma, axis=1)
    return np.concatenate([chroma_mean, chroma_var])

def extract_rms(y):
    rms = librosa.feature.rms(y=y)
    return np.array([np.mean(rms), np.var(rms)])

def extract_spectral(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    features = []
    for f in [centroid, bandwidth, rolloff, zcr]:
        features.append(np.mean(f))
        features.append(np.var(f))
    return np.array(features)

def extract_tempo(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    return np.array([tempo[0]])

def extract_harmony(y):
    y_harmonic, _ = librosa.effects.hpss(y)
    return np.array([np.mean(y_harmonic), np.var(y_harmonic)])

def extract_all_features(file_path, n_mfcc=20, sr=22050):
    features = {}

    y, sr = librosa.load(file_path, sr=sr)
    extracted_features = np.concatenate([
        extract_tempo(y, sr),
        extract_chroma(y, sr),
        extract_rms(y),
        extract_spectral(y, sr),
        extract_harmony(y),
        extract_mfcc(y, sr, n_mfcc),
    ])

    idx = 0
    for fidx in utils.RELEVANT_FEATURES:
        fname = utils.feature_idx[fidx]
        features[fname] = extracted_features[idx]
        idx += 1

    return features

def load_audio_files(audio_folder):
    samples = {}
    for filename in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.wav', '.mp3')):
            samples[filename] = extract_all_features(file_path)
            print(f'Extracted features of {filename}')
    return samples

def prediction(audio_folder, model_path='model.json'):
    print("Loading model...")
    model = load_model(model_path)
    
    print("Loading test features...")
    samples = load_audio_files(audio_folder)

    print("Predicting...")
    predictions = {}
    for name, feats in samples.items():
        predictions[name] = predict(feats, model)

    return predictions

if __name__ == '__main__':
    audio_folder = './audio'
    
    preds = prediction(audio_folder)
    fav = 0
    for name, genre in preds.items():
        print(f"{name}: {genre}")
        if name.split('.')[0] == genre:
            fav += 1
    
    print(f'Model accuracy: {fav/len(preds)}')
