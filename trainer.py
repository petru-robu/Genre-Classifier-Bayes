import csv, json, utils
from collections import defaultdict

# load raw features
def load_features(csv_path):
    features = {}
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            genre = row[utils.idx_of_feature['label']]
            if genre == 'label':
                continue
            if genre not in features:
                features[genre] = {}
            for idx in utils.RELEVANT_FEATURES:
                fname = utils.feature_idx[idx]
                if fname not in features[genre]:
                    features[genre][fname] = []
                features[genre][fname].append(float(row[idx]))
    return features

# compute feature probabilities
def compute_feature_probs(disc_features):
    feature_probs = defaultdict(lambda: defaultdict(list))
    for genre, feats in disc_features.items():
        for fname, bins in feats.items():
            counts = [0] * utils.NBINS
            for b in bins:
                counts[b] += 1
            probs = [(c + utils.SMOOTHING) / (len(bins) + utils.SMOOTHING * utils.NBINS) for c in counts]
            feature_probs[genre][fname] = probs
    return feature_probs

# compute priors
def compute_class_priors(disc_features):
    total_samples = sum(len(next(iter(feats.values()))) for feats in disc_features.values())
    return {genre: len(next(iter(feats.values()))) / total_samples for genre, feats in disc_features.items()}

if __name__ =='__main__':
    features = load_features('./data/half1.csv')
    bounds = utils.compute_bounds(features)
    disc = utils.discretize_features(features, bounds)
    feature_probs = compute_feature_probs(disc)

    class_priors = compute_class_priors(disc)

    model_data = {
        'feature_probs': feature_probs,
        'class_priors': class_priors,
        'feature_bounds': bounds
    }

    with open("model.json", "w") as f:
        json.dump(model_data, f, indent=4)
        print('Model saved in model.json!')
