import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from scipy import stats
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from prometheus_client import Counter, Gauge, Histogram
from app.database.database_manager import db_manager, UserSession, Block
from app.config import settings

# Configuración de logging
logger = logging.getLogger(__name__)

# Métricas Prometheus
BEHAVIOR_ANOMALY_SCORE = Gauge('behavior_anomaly_score', 'Anomaly score from behavioral analysis')
BEHAVIOR_PATTERN_CHANGES = Counter('behavior_pattern_changes_total', 'Total behavioral pattern changes')
HUMAN_LIKELIHOOD_SCORE = Gauge('human_likelihood_score', 'Human-like behavior score')
BEHAVIOR_TRAINING_COUNT = Counter('behavior_training_total', 'Total behavior model trainings')

@dataclass
class BehavioralConfig:
    min_confidence_threshold: float = 0.85
    max_learning_rate: float = 0.1
    pattern_history_size: int = 1000
    retrain_interval: int = 1000
    cluster_epsilon: float = 0.5
    cluster_min_samples: int = 5

class BehaviorType(Enum):
    MOUSE_MOVEMENT = "mouse_movement"
    CLICK_PATTERN = "click_pattern"
    SCROLL_BEHAVIOR = "scroll_behavior"
    TIMING_PATTERN = "timing_pattern"
    NAVIGATION_FLOW = "navigation_flow"

@dataclass
class BehavioralFeature:
    type: BehaviorType
    values: np.ndarray
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any]

class AdvancedBehavioralAnalyzer:
    def __init__(self):
        self.config = BehavioralConfig()
        self.behavior_profiles = {}
        self.pattern_history = {}
        self.models = self._initialize_models()
        self.scalers = self._initialize_scalers()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_metrics()
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Inicializar todos los modelos de ML para análisis comportamental"""
        return {
            "isolation_forest": IsolationForest(contamination=0.05, random_state=42),
            "dbscan": DBSCAN(eps=self.config.cluster_epsilon, min_samples=self.config.cluster_min_samples),
            "gaussian_mixture": GaussianMixture(n_components=3, random_state=42),
            "neural_network": self._create_neural_network(),
            "kmeans": KMeans(n_clusters=3, random_state=42)
        }
    
    def _initialize_scalers(self) -> Dict[str, Any]:
        """Inicializar scalers para diferentes tipos de comportamientos"""
        return {
            BehaviorType.MOUSE_MOVEMENT: RobustScaler(),
            BehaviorType.CLICK_PATTERN: StandardScaler(),
            BehaviorType.SCROLL_BEHAVIOR: RobustScaler(),
            BehaviorType.TIMING_PATTERN: StandardScaler(),
            BehaviorType.NAVIGATION_FLOW: StandardScaler()
        }
    
    def _create_neural_network(self) -> nn.Module:
        """Crear red neuronal para detección de patrones"""
        class BehaviorNet(nn.Module):
            def __init__(self, input_size: int, hidden_size: int = 64):
                super(BehaviorNet, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, 1)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.fc3(x))
                return x
        
        return BehaviorNet(input_size=20).to(self.device)
    
    def _setup_metrics(self):
        """Configurar optimizadores y funciones de pérdida"""
        self.optimizer = optim.Adam(self.models["neural_network"].parameters(), lr=0.001)
        self.loss_fn = nn.BCELoss()
    
    async def analyze_behavior(self, user_id: int, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar comportamiento del usuario en tiempo real"""
        try:
            # Extraer y normalizar características
            features = self._extract_features(behavior_data)
            normalized_features = self._normalize_features(features)
            
            # Realizar análisis múltiple
            analysis_results = await asyncio.gather(
                self._perform_anomaly_detection(normalized_features),
                self._perform_pattern_analysis(normalized_features),
                self._perform_cluster_analysis(normalized_features),
                self._perform_neural_analysis(normalized_features)
            )
            
            # Combinar resultados
            combined_score = self._combine_results(analysis_results)
            
            # Actualizar perfil de comportamiento
            await self._update_behavior_profile(user_id, features, combined_score)
            
            # Guardar en base de datos
            await self._save_behavior_analysis(user_id, behavior_data, combined_score)
            
            return {
                "anomaly_score": combined_score["anomaly_score"],
                "human_likelihood": combined_score["human_likelihood"],
                "pattern_consistency": combined_score["pattern_consistency"],
                "recommended_action": self._get_recommended_action(combined_score),
                "confidence": combined_score["confidence"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing behavior: {e}")
            return {
                "anomaly_score": 0.0,
                "human_likelihood": 0.0,
                "pattern_consistency": 0.0,
                "recommended_action": "continue",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _extract_features(self, behavior_data: Dict[str, Any]) -> Dict[BehaviorType, np.ndarray]:
        """Extraer características avanzadas del comportamiento"""
        features = {}
        
        # Características de movimiento del mouse
        if "mouse_movements" in behavior_data:
            mouse_features = self._extract_mouse_features(behavior_data["mouse_movements"])
            features[BehaviorType.MOUSE_MOVEMENT] = mouse_features
        
        # Características de patrones de clic
        if "click_patterns" in behavior_data:
            click_features = self._extract_click_features(behavior_data["click_patterns"])
            features[BehaviorType.CLICK_PATTERN] = click_features
        
        # Características de scroll
        if "scroll_behavior" in behavior_data:
            scroll_features = self._extract_scroll_features(behavior_data["scroll_behavior"])
            features[BehaviorType.SCROLL_BEHAVIOR] = scroll_features
        
        # Características de timing
        if "timing_data" in behavior_data:
            timing_features = self._extract_timing_features(behavior_data["timing_data"])
            features[BehaviorType.TIMING_PATTERN] = timing_features
        
        # Características de flujo de navegación
        if "navigation_flow" in behavior_data:
            nav_features = self._extract_navigation_features(behavior_data["navigation_flow"])
            features[BehaviorType.NAVIGATION_FLOW] = nav_features
        
        return features
    
    def _extract_mouse_features(self, movements: List[Dict]) -> np.ndarray:
        """Extraer características avanzadas de movimiento del mouse"""
        if not movements:
            return np.zeros(10)
        
        points = np.array([(m['x'], m['y']) for m in movements])
        
        features = [
            # Velocidad y aceleración
            np.mean([m.get('velocity', 0) for m in movements]),
            np.std([m.get('velocity', 0) for m in movements]),
            
            # Patrón de movimiento
            self._calculate_movement_entropy(points),
            self._calculate_movement_linearity(points),
            
            # Distribución espacial
            np.mean(points[:, 0]),  # mean x
            np.mean(points[:, 1]),  # mean y
            np.std(points[:, 0]),   # std x
            np.std(points[:, 1]),   # std y
            
            # Tiempos
            np.mean([m.get('duration', 0) for m in movements]),
            np.std([m.get('duration', 0) for m in movements])
        ]
        
        return np.array(features)
    
    def _extract_click_features(self, clicks: List[Dict]) -> np.ndarray:
        """Extraer características de patrones de clic"""
        if not clicks:
            return np.zeros(8)
        
        features = [
            # Frecuencia y timing
            len(clicks),
            np.mean([c.get('duration', 0) for c in clicks]),
            np.std([c.get('duration', 0) for c in clicks]),
            
            # Patrones de intervalo
            self._calculate_click_intervals(clicks),
            self._calculate_click_consistency(clicks),
            
            # Precisión
            np.mean([c.get('accuracy', 0) for c in clicks]),
            np.std([c.get('accuracy', 0) for c in clicks]),
            
            # Distribución de tipos de clic
            sum(1 for c in clicks if c.get('type') == 'left') / len(clicks)
        ]
        
        return np.array(features)
    
    def _extract_scroll_features(self, scrolls: List[Dict]) -> np.ndarray:
        """Extraer características de comportamiento de scroll"""
        if not scrolls:
            return np.zeros(6)
        
        features = [
            # Patrones de scroll
            np.mean([s.get('distance', 0) for s in scrolls]),
            np.std([s.get('distance', 0) for s in scrolls]),
            np.mean([s.get('velocity', 0) for s in scrolls]),
            np.std([s.get('velocity', 0) for s in scrolls]),
            
            # Direccionalidad
            sum(1 for s in scrolls if s.get('direction') == 'down') / len(scrolls),
            
            # Consistencia
            self._calculate_scroll_consistency(scrolls)
        ]
        
        return np.array(features)
    
    def _extract_timing_features(self, timings: Dict[str, Any]) -> np.ndarray:
        """Extraer características de patrones de timing"""
        features = [
            # Tiempos de respuesta
            timings.get('response_time_mean', 0),
            timings.get('response_time_std', 0),
            
            # Patrones de pausa
            timings.get('pause_duration_mean', 0),
            timings.get('pause_duration_std', 0),
            
            # Consistencia temporal
            timings.get('timing_consistency', 0),
            
            # Variabilidad
            timings.get('timing_variability', 0)
        ]
        
        return np.array(features)
    
    def _extract_navigation_features(self, navigation: Dict[str, Any]) -> np.ndarray:
        """Extraer características de flujo de navegación"""
        features = [
            # Patrones de navegación
            navigation.get('page_transitions', 0),
            navigation.get('unique_pages', 0),
            
            # Profundidad de navegación
            navigation.get('max_depth', 0),
            navigation.get('avg_depth', 0),
            
            # Tiempo por página
            navigation.get('avg_time_per_page', 0),
            navigation.get('time_variability', 0),
            
            # Patrones de retorno
            navigation.get('back_navigation_count', 0)
        ]
        
        return np.array(features)
    
    def _normalize_features(self, features: Dict[BehaviorType, np.ndarray]) -> Dict[BehaviorType, np.ndarray]:
        """Normalizar características usando los scalers apropiados"""
        normalized = {}
        
        for behavior_type, feature_array in features.items():
            scaler = self.scalers[behavior_type]
            
            # Ajustar scaler si es necesario
            if not hasattr(scaler, 'n_samples_seen_') or scaler.n_samples_seen_ == 0:
                scaler.fit(feature_array.reshape(1, -1))
            
            normalized[behavior_type] = scaler.transform(feature_array.reshape(1, -1))[0]
        
        return normalized
    
    async def _perform_anomaly_detection(self, features: Dict[BehaviorType, np.ndarray]) -> Dict[str, float]:
        """Detección de anomalías usando múltiples algoritmos"""
        combined_features = np.concatenate([f for f in features.values()])
        
        # Isolation Forest
        iso_score = self.models["isolation_forest"].score_samples([combined_features])[0]
        
        # DBSCAN (convertir a score)
        dbscan_labels = self.models["dbscan"].fit_predict([combined_features])
        dbscan_score = 1.0 if dbscan_labels[0] != -1 else -1.0
        
        return {
            "isolation_score": float(iso_score),
            "dbscan_score": float(dbscan_score),
            "combined_anomaly": float((iso_score + dbscan_score) / 2)
        }
    
    async def _perform_pattern_analysis(self, features: Dict[BehaviorType, np.ndarray]) -> Dict[str, float]:
        """Análisis de patrones comportamentales"""
        combined_features = np.concatenate([f for f in features.values()])
        
        # Gaussian Mixture
        gm_score = self.models["gaussian_mixture"].score_samples([combined_features])[0]
        
        # K-Means (distancia al centroide más cercano)
        kmeans = self.models["kmeans"]
        distances = kmeans.transform([combined_features])
        min_distance = np.min(distances)
        kmeans_score = np.exp(-min_distance)  # Convertir distancia a score
        
        return {
            "gaussian_score": float(gm_score),
            "kmeans_score": float(kmeans_score),
            "pattern_consistency": float((gm_score + kmeans_score) / 2)
        }
    
    async def _perform_cluster_analysis(self, features: Dict[BehaviorType, np.ndarray]) -> Dict[str, float]:
        """Análisis de clustering para detección de patrones"""
        combined_features = np.concatenate([f for f in features.values()])
        
        # Calcular similitud con clusters existentes
        cluster_similarity = self._calculate_cluster_similarity(combined_features)
        
        # Calcular densidad de puntos
        point_density = self._calculate_point_density(combined_features)
        
        return {
            "cluster_similarity": float(cluster_similarity),
            "point_density": float(point_density),
            "cluster_quality": float((cluster_similarity + point_density) / 2)
        }
    
    async def _perform_neural_analysis(self, features: Dict[BehaviorType, np.ndarray]) -> Dict[str, float]:
        """Análisis usando red neuronal"""
        combined_features = np.concatenate([f for f in features.values()])
        
        # Convertir a tensor
        features_tensor = torch.FloatTensor(combined_features).to(self.device)
        
        # Predicción
        self.models["neural_network"].eval()
        with torch.no_grad():
            prediction = self.models["neural_network"](features_tensor.unsqueeze(0))
        
        return {
            "nn_human_score": float(prediction.item()),
            "nn_confidence": float(1.0 - abs(prediction.item() - 0.5) * 2)
        }
    
    def _combine_results(self, analysis_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Combinar resultados de todos los análisis"""
        return {
            "anomaly_score": np.mean([
                analysis_results[0]["combined_anomaly"],
                1.0 - analysis_results[1]["pattern_consistency"],
                1.0 - analysis_results[2]["cluster_quality"]
            ]),
            "human_likelihood": analysis_results[3]["nn_human_score"],
            "pattern_consistency": analysis_results[1]["pattern_consistency"],
            "confidence": np.mean([
                analysis_results[3]["nn_confidence"],
                analysis_results[2]["cluster_similarity"]
            ])
        }
    
    async def _update_behavior_profile(self, user_id: int, features: Dict[BehaviorType, np.ndarray], scores: Dict[str, float]):
        """Actualizar perfil de comportamiento del usuario"""
        if user_id not in self.behavior_profiles:
            self.behavior_profiles[user_id] = {
                "behavior_history": [],
                "pattern_models": {},
                "last_updated": datetime.utcnow(),
                "anomaly_count": 0,
                "learning_rate": self.config.max_learning_rate
            }
        
        profile = self.behavior_profiles[user_id]
        
        # Agregar a historial
        behavior_record = {
            "features": {k.name: v.tolist() for k, v in features.items()},
            "scores": scores,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        profile["behavior_history"].append(behavior_record)
        
        # Mantener tamaño del historial
        if len(profile["behavior_history"]) > self.config.pattern_history_size:
            profile["behavior_history"] = profile["behavior_history"][-self.config.pattern_history_size:]
        
        # Actualizar modelos si es necesario
        if len(profile["behavior_history"]) % self.config.retrain_interval == 0:
            await self._retrain_models(user_id)
        
        # Ajustar tasa de aprendizaje basado en consistencia
        self._adjust_learning_rate(user_id, scores["pattern_consistency"])
    
    async def _retrain_models(self, user_id: int):
        """Reentrenar modelos con nuevo datos"""
        try:
            profile = self.behavior_profiles[user_id]
            history = profile["behavior_history"]
            
            if len(history) < 50:  # Mínimo de muestras para entrenar
                return
            
            # Preparar datos de entrenamiento
            X = []
            for record in history:
                features = np.concatenate([np.array(v) for v in record["features"].values()])
                X.append(features)
            
            X = np.array(X)
            
            # Reentrenar modelos
            self.models["isolation_forest"].fit(X)
            self.models["gaussian_mixture"].fit(X)
            self.models["kmeans"].fit(X)
            
            # Entrenar red neuronal
            await self._train_neural_network(X)
            
            BEHAVIOR_TRAINING_COUNT.inc()
            logger.info(f"Models retrained for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    async def _train_neural_network(self, X: np.ndarray):
        """Entrenar red neuronal con datos de comportamiento"""
        try:
            # Crear etiquetas (asumir que el comportamiento actual es humano)
            y = np.ones(len(X))
            
            # Convertir a tensores
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Entrenamiento
            self.models["neural_network"].train()
            for epoch in range(10):
                self.optimizer.zero_grad()
                outputs = self.models["neural_network"](X_tensor)
                loss = self.loss_fn(outputs.squeeze(), y_tensor)
                loss.backward()
                self.optimizer.step()
                
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
    
    def _adjust_learning_rate(self, user_id: int, consistency: float):
        """Ajustar tasa de aprendizaje basado en consistencia"""
        profile = self.behavior_profiles[user_id]
        
        # Reducir tasa de aprendizaje si el comportamiento es consistente
        if consistency > 0.8:
            profile["learning_rate"] = max(0.01, profile["learning_rate"] * 0.9)
        else:
            profile["learning_rate"] = min(
                self.config.max_learning_rate,
                profile["learning_rate"] * 1.1
            )
    
    async def _save_behavior_analysis(self, user_id: int, raw_data: Dict[str, Any], scores: Dict[str, float]):
        """Guardar análisis en base de datos"""
        try:
            with db_manager.get_db() as db:
                # Crear registro de análisis
                analysis_record = {
                    "user_id": user_id,
                    "raw_data": json.dumps(raw_data),
                    "anomaly_score": scores["anomaly_score"],
                    "human_likelihood": scores["human_likelihood"],
                    "pattern_consistency": scores["pattern_consistency"],
                    "confidence": scores["confidence"],
                    "created_at": datetime.utcnow()
                }
                
                # Aquí iría el código para guardar en tu modelo de base de datos
                # Por ejemplo: db.add(BehaviorAnalysis(**analysis_record))
                # db.commit()
                
                pass
                
        except Exception as e:
            logger.error(f"Error saving behavior analysis: {e}")
    
    def _get_recommended_action(self, scores: Dict[str, float]) -> str:
        """Determinar acción recomendada basada en los scores"""
        if scores["anomaly_score"] > 0.8:
            return "block"
        elif scores["anomaly_score"] > 0.6:
            return "challenge"
        elif scores["human_likelihood"] < 0.3:
            return "monitor"
        else:
            return "allow"
    
    # Métodos de utilidad para cálculo de características
    def _calculate_movement_entropy(self, points: np.ndarray) -> float:
        """Calcular entropía del movimiento del mouse"""
        if len(points) < 2:
            return 0.0
        
        # Calcular ángulos entre movimientos
        vectors = np.diff(points, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # Calcular entropía de los ángulos
        hist, _ = np.histogram(angles, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))
    
    def _calculate_movement_linearity(self, points: np.ndarray) -> float:
        """Calcular linearidad del movimiento"""
        if len(points) < 3:
            return 0.0
        
        # Ajustar línea recta a los puntos
        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calcular R²
        y_pred = m * x + c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1.0 - (ss_res / (ss_tot + 1e-10))
    
    def _calculate_click_intervals(self, clicks: List[Dict]) -> float:
        """Calcular consistencia de intervalos entre clics"""
        if len(clicks) < 2:
            return 0.0
        
        timestamps = [c['timestamp'] for c in clicks]
        intervals = np.diff(timestamps)
        
        return float(np.std(intervals) / (np.mean(intervals) + 1e-10))
    
    def _calculate_click_consistency(self, clicks: List[Dict]) -> float:
        """Calcular consistencia de patrones de clic"""
        if len(clicks) < 3:
            return 0.0
        
        # Calcular variabilidad en duración y precisión
        durations = [c.get('duration', 0) for c in clicks]
        accuracies = [c.get('accuracy', 0) for c in clicks]
        
        duration_cv = np.std(durations) / (np.mean(durations) + 1e-10)
        accuracy_cv = np.std(accuracies) / (np.mean(accuracies) + 1e-10)
        
        return 1.0 - (duration_cv + accuracy_cv) / 2
    
    def _calculate_scroll_consistency(self, scrolls: List[Dict]) -> float:
        """Calcular consistencia de patrones de scroll"""
        if len(scrolls) < 3:
            return 0.0
        
        distances = [s.get('distance', 0) for s in scrolls]
        velocities = [s.get('velocity', 0) for s in scrolls]
        
        distance_cv = np.std(distances) / (np.mean(distances) + 1e-10)
        velocity_cv = np.std(velocities) / (np.mean(velocities) + 1e-10)
        
        return 1.0 - (distance_cv + velocity_cv) / 2
    
    def _calculate_cluster_similarity(self, features: np.ndarray) -> float:
        """Calcular similitud con clusters existentes"""
        try:
            # Usar K-Means para calcular distancia al centroide más cercano
            kmeans = self.models["kmeans"]
            distances = kmeans.transform(features.reshape(1, -1))
            min_distance = np.min(distances)
            
            # Convertir distancia a similitud (0-1)
            return float(np.exp(-min_distance))
        except:
            return 0.5
    
    def _calculate_point_density(self, features: np.ndarray) -> float:
        """Calcular densidad de puntos alrededor del features actual"""
        try:
            # Usar DBSCAN para estimar densidad
            dbscan = self.models["dbscan"]
            labels = dbscan.fit_predict(features.reshape(1, -1))
            
            # Si el punto no es outlier, alta densidad
            return 1.0 if labels[0] != -1 else 0.0
        except:
            return 0.5
    
    async def get_behavior_summary(self, user_id: int) -> Dict[str, Any]:
        """Obtener resumen del comportamiento del usuario"""
        if user_id not in self.behavior_profiles:
            return {"error": "No behavior data available"}
        
        profile = self.behavior_profiles[user_id]
        history = profile["behavior_history"]
        
        if not history:
            return {"error": "No behavior history"}
        
        # Calcular estadísticas
        anomaly_scores = [r["scores"]["anomaly_score"] for r in history]
        human_scores = [r["scores"]["human_likelihood"] for r in history]
        consistency_scores = [r["scores"]["pattern_consistency"] for r in history]
        
        return {
            "total_samples": len(history),
            "avg_anomaly_score": float(np.mean(anomaly_scores)),
            "avg_human_score": float(np.mean(human_scores)),
            "avg_consistency": float(np.mean(consistency_scores)),
            "last_updated": profile["last_updated"].isoformat(),
            "learning_rate": profile["learning_rate"],
            "anomaly_count": profile["anomaly_count"],
            "behavior_trend": self._calculate_behavior_trend(history)
        }
    
    def _calculate_behavior_trend(self, history: List[Dict]) -> str:
        """Calcular tendencia del comportamiento"""
        if len(history) < 10:
            return "insufficient_data"
        
        # Obtener scores de las últimas 10 muestras
        recent_scores = [r["scores"]["anomaly_score"] for r in history[-10:]]
        
        # Calcular tendencia
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.01:
            return "worsening"
        elif trend < -0.01:
            return "improving"
        else:
            return "stable"

# Instancia global del analizador de comportamiento
behavior_analyzer = AdvancedBehavioralAnalyzer()

# Funciones de utilidad para uso en otras partes del sistema
async def analyze_user_behavior(user_id: int, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
    """Función conveniente para análisis de comportamiento"""
    return await behavior_analyzer.analyze_behavior(user_id, behavior_data)

async def get_behavior_report(user_id: int) -> Dict[str, Any]:
    """Obtener reporte de comportamiento del usuario"""
    return await behavior_analyzer.get_behavior_summary(user_id)

# Inicialización asíncrona
async def initialize_behavior_analyzer():
    """Inicializar el analizador de comportamiento"""
    # Aquí puedes cargar datos iniciales o modelos pre-entrenados
    logger.info("Behavior analyzer initialized successfully")