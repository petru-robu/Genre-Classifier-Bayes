import json

FEATURES_STR = """filename,length,chroma_stft_mean,chroma_stft_var,
rms_mean,rms_var,spectral_centroid_mean,spectral_centroid_var,
spectral_bandwidth_mean,spectral_bandwidth_var,rolloff_mean,rolloff_var,
zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,
perceptr_mean,perceptr_var,tempo,mfcc1_mean,mfcc1_var,mfcc2_mean,mfcc2_var,
mfcc3_mean,mfcc3_var,mfcc4_mean,mfcc4_var,mfcc5_mean,mfcc5_var,mfcc6_mean,
mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,mfcc8_var,mfcc9_mean,mfcc9_var,
mfcc10_mean,mfcc10_var,mfcc11_mean,mfcc11_var,mfcc12_mean,mfcc12_var,mfcc13_mean,
mfcc13_var,mfcc14_mean,mfcc14_var,mfcc15_mean,mfcc15_var,mfcc16_mean,mfcc16_var,
mfcc17_mean,mfcc17_var,mfcc18_mean,mfcc18_var,mfcc19_mean,mfcc19_var,mfcc20_mean,
mfcc20_var,label"""

NBINS = 20
SMOOTHING = 1

feature_idx = {i: f.strip() for i, f in enumerate(FEATURES_STR.split(','))}
idx_of_feature = {f: i for i, f in feature_idx.items()}

RELEVANT_FEATURES = []
RELEVANT_FEATURES.append(idx_of_feature['tempo'])
RELEVANT_FEATURES.append(idx_of_feature['chroma_stft_mean'])
RELEVANT_FEATURES.append(idx_of_feature['chroma_stft_var'])
RELEVANT_FEATURES.append(idx_of_feature['rms_mean'])
RELEVANT_FEATURES.append(idx_of_feature['rms_var'])
RELEVANT_FEATURES.append(idx_of_feature['spectral_centroid_mean'])
RELEVANT_FEATURES.append(idx_of_feature['spectral_centroid_var'])
RELEVANT_FEATURES.append(idx_of_feature['spectral_bandwidth_mean'])
RELEVANT_FEATURES.append(idx_of_feature['spectral_bandwidth_var'])
# RELEVANT_FEATURES.append(idx_of_feature['rolloff_mean'])
# RELEVANT_FEATURES.append(idx_of_feature['rolloff_var'])
# RELEVANT_FEATURES.append(idx_of_feature['zero_crossing_rate_mean'])
# RELEVANT_FEATURES.append(idx_of_feature['zero_crossing_rate_var'])
RELEVANT_FEATURES.append(idx_of_feature['harmony_mean'])
RELEVANT_FEATURES.append(idx_of_feature['harmony_var'])
# RELEVANT_FEATURES.append(idx_of_feature['perceptr_mean'])
# RELEVANT_FEATURES.append(idx_of_feature['perceptr_var'])
for i in range(1, 21):
    RELEVANT_FEATURES.append(idx_of_feature[f'mfcc{i}_mean'])
    RELEVANT_FEATURES.append(idx_of_feature[f'mfcc{i}_var'])


# compute feature bounds
def compute_bounds(features):
    bounds = {}
    for idx in RELEVANT_FEATURES:
        fname = feature_idx[idx]
        min_val, max_val = float('inf'), float('-inf')
        for genre in features:
            l, r = min(features[genre][fname]), max(features[genre][fname])
            min_val = min(min_val, l)
            max_val = max(max_val, r)
        bounds[fname] = (min_val, max_val)
    return bounds

# discretize - put a float value in a bin
def discretize(value, bounds, nbins=NBINS):
    left, right = bounds
    if value <= left:
        return 0
    if value >= right:
        return nbins - 1
    bin_width = (right - left) / nbins
    return min(int((value - left) / bin_width), nbins - 1)

# discretize features
def discretize_features(features, bounds):
    disc_features = {}
    for genre, feats in features.items():
        disc_features[genre] = {}
        for fname, vals in feats.items():
            disc_features[genre][fname] = [discretize(v, bounds[fname]) for v in vals]
    return disc_features

# dump dict 
def dump_dict(dict, dict_name):
    with open(f'./debug/{dict_name}.json', 'w') as f:
        json.dump(dict, f, indent=4)
