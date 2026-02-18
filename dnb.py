"""
Implement discrete Naive Bayes.
"""

import numpy as np

class DiscreteNB:
    def __init__(self, n_bins=10, smoothing=1.0):
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.classes = None
        self.feature_probs = None
        self.class_priors = None
        self.feature_bounds = None

    # Fit the model
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.feature_bounds = np.zeros((n_features, 2))  # min/max per feature

        # compute feature bounds for discretization
        for i in range(n_features):
            self.feature_bounds[i, 0] = X[:, i].min()
            self.feature_bounds[i, 1] = X[:, i].max()

        # discretize features
        X_disc = self._discretize(X)

        # compute class priors
        self.class_priors = {cl: (y == cl).sum() / len(y)
                             for cl in self.classes}

        # compute feature probabilities per class and bin
        self.feature_probs = {cl: np.zeros(
            (n_features, self.n_bins)) for cl in self.classes}

        for cl in self.classes:
            X_c = X_disc[y == cl]
            for i in range(n_features):
                counts = np.zeros(self.n_bins)
                for b in X_c[:, i]:
                    counts[int(b)] += 1
                probs = (counts + self.smoothing) / (counts.sum() + self.smoothing * self.n_bins)
                self.feature_probs[cl][i, :] = probs

    # Discretize raw feature values into bins
    def _discretize(self, X):
        X_disc = np.zeros_like(X, dtype=int)
        for i in range(X.shape[1]):
            fmin, fmax = self.feature_bounds[i]
            if fmax == fmin:
                X_disc[:, i] = 0
            else:
                bins = ((X[:, i] - fmin) / (fmax - fmin) * self.n_bins).astype(int)
                bins = np.clip(bins, 0, self.n_bins - 1)
                X_disc[:, i] = bins
        return X_disc

    # Predict for new data
    def predict(self, X):
        X_disc = self._discretize(X)
        y_pred = []

        for x in X_disc:
            log_posteriors = {}
            for cl in self.classes:
                log_prior = np.log(self.class_priors[cl])
                log_likelihood = 0
                for i, b in enumerate(x):
                    log_likelihood += np.log(self.feature_probs[cl][i, b])
                log_posteriors[cl] = log_prior + log_likelihood
            y_pred.append(max(log_posteriors, key=log_posteriors.get))
        return np.array(y_pred)
