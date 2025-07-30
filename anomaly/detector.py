import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from .features import FeatureExtractor

class detector:
    def __init__(self, n_components=10, contamination=0.01):
        self.detector = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        self.pca = PCA(n_components=n_components)
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False

    def extract_features(self, images):
        return self.feature_extractor.extract(images)

    def train(self, images):
        features = np.vstack([self.extract_features(image) for image in images])
        reduced = self.pca.fit_transform(features)
        self.detector.fit(reduced)
        self.is_trained = True

    def detect(self, images):
        if not self.is_trained:
            raise ValueError("Detector must be trained before use")
        features = self.extract_features(images)
        reduced = self.pca.transform(features.reshape(1, -1))
        return self.detector.decision_function(reduced)[0]

    