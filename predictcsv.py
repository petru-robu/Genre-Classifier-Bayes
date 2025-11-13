import csv, json, utils
from predictor import predict, load_model

# load samples
def load_test_features(csv_path):
    samples = {}
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[utils.idx_of_feature['label']] == 'label':
                continue
            sample_name = row[utils.idx_of_feature['filename']]
            samples[sample_name] = {}
            for idx in utils.RELEVANT_FEATURES:
                fname = utils.feature_idx[idx]
                if fname == 'label':
                    continue
                samples[sample_name][fname] = float(row[idx])
    return samples

def prediction(csv_path='data/half2.csv', model_path='model.json'):
    print("Loading model...")
    model = load_model(model_path)
    
    print("Loading test features...")
    samples = load_test_features(csv_path)

    print(samples['rock.00098.wav'])

    print("Predicting...")
    predictions = {}
    for name, feats in samples.items():
        #print(name, feats)
        predictions[name] = predict(feats, model)

    return predictions

if __name__ == "__main__":
    preds = prediction()
    fav = 0
    for name, genre in preds.items():
        print(f"{name}: {genre}")
        if name.split('.')[0] == genre:
            fav += 1
    
    print(f'Model accuracy: {fav/len(preds)}')