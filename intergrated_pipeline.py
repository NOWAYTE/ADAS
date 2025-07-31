import time
from preprocessing import preprocessor
from anomaly import detector
from XAI import explainer

class pipeline:
    def __init__(self, config):
        self.preprocessor = preprocessor(config)
        self.detector = detector(config)
        self.explainer = explainer(config)
        self.frame_count = 0
        self.anomaly_scores = []

    def process(self, image):
        start_time = time.time()
        processed = self.preprocessor.process(image)
        preprocess_time = time.time() - start_time
        score, is_anomaly = self.detector.detect(processed)
        detect_time = time.time() - start_time
        explanation = self.explainer.explain(processed)
        self.frame_count += 1
        self.anomaly_scores.append(score)

        return {
            'frame': processed,
            'processed_time': {
               'preprocess_ms': preprocess_time * 1000,
                'detect_ms': detect_time * 1000,
                'total_ms': (time.time() - start_time) * 1000
            },
            'score': score,
            'is_anomaly': is_anomaly,
            'explanation': explanation
        }
    def train_detector(self, clean_frames: list):
        processed = [self.preprocessor.process(frame) for frame in clean_frames]
        self.detector.train(processed)