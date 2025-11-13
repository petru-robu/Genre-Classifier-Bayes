import json, math, utils

# load model
def load_model(path='model.json'):
    with open(path) as f:
        return json.load(f)

def predict(sample, model):
    feature_probs = model['feature_probs']
    class_priors = model['class_priors']
    feature_bounds = model['feature_bounds']
    
    scores = {}
    for genre in feature_probs:
        score = math.log(class_priors[genre])

        for feature_name, value in sample.items():
            bin_val = utils.discretize(value, feature_bounds[feature_name], utils.NBINS)            
            prob = feature_probs[genre][feature_name][bin_val]
            score += math.log(prob)

        scores[genre] = score

    predicted_genre = max(scores, key=scores.get)
    return predicted_genre