import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
import joblib
from typing import Dict, List, Tuple
import asyncio
import logging
from prometheus_client import Counter, Gauge, Histogram
import os
from datetime import datetime
import hashlib
import random
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

ANOMALY_SCORE = Gauge('anomaly_score', 'Anomaly score from ML model')
EVASION_TRIGGERED = Counter('evasion_triggered_total', 'Total evasion triggers')
MODEL_UPDATE_COUNT = Counter('model_update_total', 'Total model updates')
FEATURE_DRIFT_DETECTED = Counter('feature_drift_total', 'Feature drift detected')

class EvasionStrategy(Enum):
    ROTATE_FINGERPRINT = 1
    CHANGE_NETWORK = 2
    MODIFY_TEMPORAL = 3
    BEHAVIOR_MIMICRY = 4
    FULL_SESSION_RESET = 5

@dataclass
class EvasionConfig:
    min_anomaly_threshold: float = -0.7
    max_anomaly_threshold: float = -0.3
    adaptive_learning_rate: float = 0.1
    model_retrain_interval: int = 100
    feature_validation_bounds: Dict[str, Tuple[float, float]] = None

class AdvancedBehavioralMLDetector:
    def __init__(self, model_dir: str = '/models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.config = EvasionConfig(
            feature_validation_bounds={
                'click_intervals_std': (0, 10),
                'mouse_velocity': (0, 500),
                'scroll_variance': (0, 1000),
                'request_timing': (0, 5),
                'action_entropy': (0, 10)
            }
        )
        
        self.models = self.initialize_models()
        self.behavior_profiles = {}
        self.online_learning_data = []
        self.anomaly_threshold = -0.5
        self.feature_scalers = {}
        self.initialize_feature_scalers()
        self.drift_detector = self.initialize_drift_detector()
        self.evasion_history = []
        
    def initialize_models(self) -> Dict:
        models = {
            'isolation_forest': self.load_or_create_model(
                'isolation_forest', IsolationForest(contamination=0.01, random_state=42)
            ),
            'one_class_svm': self.load_or_create_model(
                'one_class_svm', OneClassSVM(nu=0.01, kernel='rbf')
            ),
            'elliptic_envelope': self.load_or_create_model(
                'elliptic_envelope', EllipticEnvelope(contamination=0.01)
            ),
            'random_forest': self.load_or_create_model(
                'random_forest', RandomForestClassifier(n_estimators=100)
            )
        }
        return models
    
    def initialize_drift_detector(self):
        from alibi_detect.cd import KSDrift
        return KSDrift(p_val=0.05)
    
    def load_or_create_model(self, model_name: str, default_model):
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        try:
            model = joblib.load(model_path)
            logger.info(f"Model {model_name} loaded from {model_path}")
            return model
        except FileNotFoundError:
            logger.warning(f"Model {model_name} not found, creating new")
            joblib.dump(default_model, model_path)
            return default_model
    
    def initialize_feature_scalers(self):
        for feature in self.config.feature_validation_bounds.keys():
            self.feature_scalers[feature] = RobustScaler()
    
    async def analyze_behavior_pattern(self, session_data: Dict) -> Tuple[float, List[EvasionStrategy]]:
        try:
            features = self.extract_and_validate_features(session_data)
            
            if self.detect_feature_drift(features):
                FEATURE_DRIFT_DETECTED.inc()
                logger.warning("Feature drift detected, recalibrating models")
                await self.recalibrate_models()
            
            normalized_features = self.normalize_features(features)
            
            anomaly_scores = []
            for model_name, model in self.models.items():
                if hasattr(model, 'decision_function'):
                    score = model.decision_function([normalized_features])[0]
                    anomaly_scores.append(score)
                elif hasattr(model, 'predict_proba'):
                    score = model.predict_proba([normalized_features])[0][1]
                    anomaly_scores.append(score)
            
            ensemble_score = np.average(anomaly_scores, weights=[0.4, 0.3, 0.2, 0.1])
            ANOMALY_SCORE.set(ensemble_score)
            
            self.dynamic_threshold_adjustment(ensemble_score)
            
            self.online_learning_data.append(normalized_features)
            if len(self.online_learning_data) >= self.config.model_retrain_interval:
                await self.update_models()
            
            evasion_strategies = []
            if ensemble_score < self.anomaly_threshold:
                EVASION_TRIGGERED.inc()
                evasion_strategies = self.determine_evasion_strategies(
                    ensemble_score, session_data)
                await self.execute_evasion_strategies(evasion_strategies, session_data)
            
            return ensemble_score, evasion_strategies
            
        except Exception as e:
            logger.error(f"Error analyzing behavior pattern: {e}")
            return 0.0, []
    
    def extract_and_validate_features(self, session_data: Dict) -> np.array:
        features = []
        for feature_name, (min_val, max_val) in self.config.feature_validation_bounds.items():
            value = session_data.get(feature_name, 0)
            clamped_value = max(min_val, min(max_val, value))
            features.append(clamped_value)
        
        return np.array(features).reshape(1, -1)
    
    def detect_feature_drift(self, features: np.array) -> bool:
        try:
            features_2d = features.reshape(1, -1)
            drift_result = self.drift_detector.predict(features_2d)
            return drift_result['data']['is_drift'] == 1
        except Exception as e:
            logger.error(f"Error detecting feature drift: {e}")
            return False
    
    def normalize_features(self, features: np.array) -> np.array:
        normalized = []
        for i, feature_value in enumerate(features[0]):
            feature_name = list(self.config.feature_validation_bounds.keys())[i]
            scaler = self.feature_scalers[feature_name]
            
            if not hasattr(scaler, 'center_'):
                sample_data = np.array([[feature_value]])
                scaler.fit(sample_data)
            
            normalized_value = scaler.transform([[feature_value]])[0][0]
            normalized.append(normalized_value)
        
        return np.array(normalized).reshape(1, -1)
    
    def dynamic_threshold_adjustment(self, current_score: float):
        self.anomaly_threshold = (self.config.adaptive_learning_rate * current_score + 
                                 (1 - self.config.adaptive_learning_rate) * self.anomaly_threshold)
        
        self.anomaly_threshold = max(self.config.min_anomaly_threshold, 
                                    min(self.config.max_anomaly_threshold, self.anomaly_threshold))
    
    async def update_models(self):
        try:
            X = np.vstack(self.online_learning_data)
            
            for model_name, model in self.models.items():
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X)
                else:
                    model.fit(X)
            
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
                joblib.dump(model, model_path)
            
            MODEL_UPDATE_COUNT.inc()
            logger.info("All models updated and saved")
            self.online_learning_data = []
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
    
    async def recalibrate_models(self):
        logger.info("Starting model recalibration due to feature drift")
        
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                os.remove(model_path)
        
        self.models = self.initialize_models()
        self.online_learning_data = []
        logger.info("Model recalibration completed")
    
    def determine_evasion_strategies(self, anomaly_score: float, 
                                   session_data: Dict) -> List[EvasionStrategy]:
        strategies = []
        
        if anomaly_score < -1.0:
            strategies.extend([
                EvasionStrategy.FULL_SESSION_RESET,
                EvasionStrategy.CHANGE_NETWORK,
                EvasionStrategy.ROTATE_FINGERPRINT
            ])
        elif anomaly_score < -0.7:
            strategies.extend([
                EvasionStrategy.CHANGE_NETWORK,
                EvasionStrategy.ROTATE_FINGERPRINT
            ])
        elif anomaly_score < -0.5:
            strategies.append(EvasionStrategy.ROTATE_FINGERPRINT)
        
        if session_data.get('consecutive_failures', 0) > 3:
            strategies.append(EvasionStrategy.MODIFY_TEMPORAL)
        
        if session_data.get('suspicious_patterns', []):
            strategies.append(EvasionStrategy.BEHAVIOR_MIMICRY)
        
        return list(set(strategies))
    
    async def execute_evasion_strategies(self, strategies: List[EvasionStrategy], 
                                       session_data: Dict):
        strategy_handlers = {
            EvasionStrategy.ROTATE_FINGERPRINT: self.rotate_digital_fingerprint,
            EvasionStrategy.CHANGE_NETWORK: self.change_network_infrastructure,
            EvasionStrategy.MODIFY_TEMPORAL: self.modify_temporal_patterns,
            EvasionStrategy.BEHAVIOR_MIMICRY: self.implement_behavioral_mimicry,
            EvasionStrategy.FULL_SESSION_RESET: self.full_session_reset
        }
        
        for strategy in strategies:
            if strategy in strategy_handlers:
                try:
                    await strategy_handlers[strategy](session_data)
                    logger.info(f"Executed evasion strategy: {strategy.name}")
                except Exception as e:
                    logger.error(f"Error executing strategy {strategy.name}: {e}")
        
        self.evasion_history.append({
            'timestamp': datetime.utcnow(),
            'strategies': [s.name for s in strategies],
            'anomaly_score': session_data.get('anomaly_score', 0),
            'session_id': session_data.get('session_id', 'unknown')
        })
    
    async def rotate_digital_fingerprint(self, session_data: Dict):
        new_fingerprint = {
            'canvas_hash': self.generate_advanced_canvas_hash(),
            'webgl_vendor': random.choice(self.get_webgl_vendors()),
            'webgl_renderer': random.choice(self.get_webgl_renderers()),
            'audio_context_hash': self.generate_audio_hash(),
            'fonts_hash': self.generate_fonts_hash(session_data['location']),
            'hardware_concurrency': random.choice([4, 8, 16]),
            'device_memory': random.choice([4, 8, 16]),
            'timezone': self.select_plausible_timezone(session_data['location']),
            'screen_resolution': self.generate_plausible_resolution(),
            'language': self.select_language(session_data['location']),
            'platform': random.choice(['Win32', 'Linux x86_64', 'MacIntel'])
        }
        
        await self.apply_advanced_fingerprint(new_fingerprint, session_data)
    
    async def change_network_infrastructure(self, session_data: Dict):
        pass
    
    async def modify_temporal_patterns(self, session_data: Dict):
        pass
    
    async def implement_behavioral_mimicry(self, session_data: Dict):
        pass
    
    async def full_session_reset(self, session_data: Dict):
        pass
    
    def generate_advanced_canvas_hash(self) -> str:
        noise = os.urandom(32)
        return hashlib.sha3_256(noise).hexdigest()
    
    def get_webgl_vendors(self) -> List[str]:
        return ['Google Inc.', 'Intel Inc.', 'NVIDIA Corporation', 'AMD']
    
    def get_webgl_renderers(self) -> List[str]:
        return ['Intel Iris OpenGL Engine', 'ANGLE (Intel, Intel(R) UHD Graphics', 
                'NVIDIA GeForce GTX 1060', 'AMD Radeon Pro 5600M']
    
    def generate_plausible_resolution(self) -> str:
        resolutions = ['1920x1080', '2560x1440', '3840x2160', '1366x768']
        return random.choice(resolutions)
    
    def select_language(self, location: str) -> str:
        lang_map = {
            'US': 'en-US',
            'UK': 'en-GB',
            'DE': 'de-DE',
            'FR': 'fr-FR',
            'ES': 'es-ES'
        }
        return lang_map.get(location, 'en-US')