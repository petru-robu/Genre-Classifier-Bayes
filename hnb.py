"""
Implement hybrid Naive Bayes.
"""

import numpy as np

class HybridNB:
    def __init__(self, n_bins=10, smoothing=1.0, skew_threshold=1.0):
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.skew_threshold = skew_threshold

        self.classes = None
        self.class_priors = None

        self.gaussian_features_idx = []
        self.gaussian_params = {}

        self.discrete_features_idx = []
        self.feature_bounds = None
        self.feature_probs = {} 

    # Fit the model
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.feature_bounds = np.zeros((n_features, 2))

        # compute skew for each feature
        feature_skew = np.array([((X[:, i] - X[:, i].mean())**3).mean() / (X[:, i].std()**3 + 1e-10)
                                 for i in range(n_features)])

        # Decide feature type
        self.gaussian_features_idx = [i for i, s in enumerate(feature_skew) if abs(s) <= self.skew_threshold]
        self.discrete_features_idx = [i for i, s in enumerate(feature_skew) if abs(s) > self.skew_threshold]

        # Compute bounds for discrete features
        for i in self.discrete_features_idx:
            self.feature_bounds[i, 0] = X[:, i].min()
            self.feature_bounds[i, 1] = X[:, i].max()

        # Compute class priors
        self.class_priors = {cl: (y == cl).sum() / n_samples for cl in self.classes}

        # Gaussian params
        for cl in self.classes:
            X_c = X[y == cl]
            if self.gaussian_features_idx:
                mean = X_c[:, self.gaussian_features_idx].mean(axis=0)
                var = X_c[:, self.gaussian_features_idx].var(axis=0) + 1e-6
                self.gaussian_params[cl] = (mean, var)

        # Discretize and compute probabilities
        for cl in self.classes:
            X_c = X[y == cl]
            if self.discrete_features_idx:
                n_disc = len(self.discrete_features_idx)
                probs = np.zeros((n_disc, self.n_bins))
                for j, i in enumerate(self.discrete_features_idx):
                    # discretize feature
                    fmin, fmax = self.feature_bounds[i]
                    bins = np.zeros(X_c.shape[0], dtype=int)
                    if fmax == fmin:
                        bins[:] = 0
                    else:
                        bins = ((X_c[:, i] - fmin) / (fmax - fmin) * self.n_bins).astype(int)
                        bins = np.clip(bins, 0, self.n_bins - 1)
                    # count occurrences
                    counts = np.bincount(bins, minlength=self.n_bins)
                    probs[j, :] = (counts + self.smoothing) / (counts.sum() + self.smoothing * self.n_bins)
                self.feature_probs[cl] = probs

    def predict(self, X):
        y_pred = []
        for x in X:
            log_posteriors = {}
            for cl in self.classes:
                log_prob = np.log(self.class_priors[cl])

                # Gaussian part
                if self.gaussian_features_idx:
                    mean, var = self.gaussian_params[cl]
                    x_g = x[self.gaussian_features_idx]
                    log_prob += -0.5 * np.sum(np.log(2 * np.pi * var) + ((x_g - mean) ** 2) / var)

                # Discrete part
                if self.discrete_features_idx:
                    probs = self.feature_probs[cl]
                    for j, i in enumerate(self.discrete_features_idx):
                        fmin, fmax = self.feature_bounds[i]
                        b = 0 if fmax == fmin else int((x[i] - fmin) / (fmax - fmin) * self.n_bins)
                        b = np.clip(b, 0, self.n_bins - 1)
                        log_prob += np.log(probs[j, b])
                log_posteriors[cl] = log_prob
            y_pred.append(max(log_posteriors, key=log_posteriors.get))
        return np.array(y_pred)


