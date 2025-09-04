from prometheus_api_client import PrometheusConnect
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from enum import Enum
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class RemediationAction(Enum):
    SCALE_UP = 1
    SCALE_DOWN = 2
    RESTART_PODS = 3
    ROTATE_PROXIES = 4
    SWITCH_REGION = 5
    ACTIVATE_STEALTH = 6
    NOTIFY_ADMINS = 7

@dataclass
class Incident:
    id: str
    severity: IncidentSeverity
    description: str
    detected_at: datetime
    metrics: Dict[str, float]
    remediation_actions: List[RemediationAction]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AdvancedAIOpsMonitoring:
    def __init__(self, prometheus_url: str, slack_webhook: str = None):
        self.prom = PrometheusConnect(url=prometheus_url)
        self.slack_client = WebClient(token=slack_webhook) if slack_webhook else None
        self.anomaly_detector = self.initialize_anomaly_detector()
        self.incident_history = []
        self.metric_baselines = self.initialize_baselines()
        self.setup_adaptive_thresholds()
        
    def initialize_anomaly_detector(self):
        return {
            'isolation_forest': IsolationForest(contamination=0.05),
            'random_forest': RandomForestClassifier(n_estimators=50),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
    
    def initialize_baselines(self) -> Dict[str, Dict]:
        return {
            'request_success_rate': {'mean': 0.95, 'std': 0.05},
            'response_time_p95': {'mean': 1500, 'std': 500},
            'proxy_failure_rate': {'mean': 0.1, 'std': 0.08},
            'capture_success_rate': {'mean': 0.85, 'std': 0.1},
            'concurrent_sessions': {'mean': 50, 'std': 20}
        }
    
    def setup_adaptive_thresholds(self):
        self.metric_thresholds = {
            metric: {
                'warning': baseline['mean'] - baseline['std'],
                'critical': baseline['mean'] - 2 * baseline['std']
            }
            for metric, baseline in self.metric_baselines.items()
        }
    
    async def continuous_health_check(self):
        while True:
            try:
                metrics = await self.collect_system_metrics()
                
                anomalies = self.detect_anomalies(metrics)
                
                if anomalies:
                    incident = self.create_incident(anomalies, metrics)
                    self.incident_history.append(incident)
                    
                    await self.auto_remediate(incident)
                    await self.notify_incident(incident)
                
                self.adjust_thresholds(metrics)
                
                self.update_anomaly_detectors(metrics)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        metrics = {}
        
        queries = {
            'request_success_rate': 'rate(http_requests_total{status=~"2.."}[5m]) / rate(http_requests_total[5m])',
            'response_time_p95': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
            'proxy_failure_rate': 'rate(proxy_failures_total[5m]) / rate(proxy_requests_total[5m])',
            'capture_success_rate': 'rate(capture_success_total[5m]) / rate(capture_attempts_total[5m])',
            'concurrent_sessions': 'sum by (instance) (sessions_active)'
        }
        
        for metric_name, query in queries.items():
            try:
                result = self.prom.custom_query(query)
                if result:
                    metrics[metric_name] = float(result[0]['value'][1])
            except Exception as e:
                logger.error(f"Error querying metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        return metrics
    
    def detect_anomalies(self, metrics: Dict[str, float]) -> List[Dict]:
        anomalies = []
        
        for metric_name, value in metrics.items():
            if metric_name in self.metric_thresholds:
                thresholds = self.metric_thresholds[metric_name]
                
                if value < thresholds['critical']:
                    anomalies.append({
                        'metric': metric_name,
                        'value': value,
                        'severity': IncidentSeverity.CRITICAL,
                        'threshold': thresholds['critical']
                    })
                elif value < thresholds['warning']:
                    anomalies.append({
                        'metric': metric_name,
                        'value': value,
                        'severity': IncidentSeverity.HIGH,
                        'threshold': thresholds['warning']
                    })
        
        ml_anomalies = self.detect_ml_anomalies(metrics)
        anomalies.extend(ml_anomalies)
        
        return anomalies
    
    def detect_ml_anomalies(self, metrics: Dict[str, float]) -> List[Dict]:
        ml_anomalies = []
        
        features = np.array([list(metrics.values())]).reshape(1, -1)
        
        iso_forest_pred = self.anomaly_detector['isolation_forest'].fit_predict(features)
        if iso_forest_pred[0] == -1:
            ml_anomalies.append({
                'metric': 'multivariate_anomaly',
                'value': 0.0,
                'severity': IncidentSeverity.MEDIUM,
                'algorithm': 'IsolationForest'
            })
        
        dbscan_pred = self.anomaly_detector['dbscan'].fit_predict(features)
        if dbscan_pred[0] == -1:
            ml_anomalies.append({
                'metric': 'cluster_anomaly',
                'value': 0.0,
                'severity': IncidentSeverity.MEDIUM,
                'algorithm': 'DBSCAN'
            })
        
        return ml_anomalies
    
    def create_incident(self, anomalies: List[Dict], metrics: Dict) -> Incident:
        max_severity = max(anomaly.get('severity', IncidentSeverity.LOW) for anomaly in anomalies)
        
        remediation_actions = self.determine_remediation_actions(anomalies)
        
        return Incident(
            id=f"incident_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            severity=max_severity,
            description=f"System anomalies detected: {[a['metric'] for a in anomalies]}",
            detected_at=datetime.utcnow(),
            metrics=metrics,
            remediation_actions=remediation_actions
        )
    
    def determine_remediation_actions(self, anomalies: List[Dict]) -> List[RemediationAction]:
        actions = []
        
        for anomaly in anomalies:
            metric = anomaly['metric']
            severity = anomaly['severity']
            
            if metric == 'request_success_rate' and severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
                actions.extend([RemediationAction.ROTATE_PROXIES, RemediationAction.ACTIVATE_STEALTH])
            
            if metric == 'response_time_p95' and severity == IncidentSeverity.CRITICAL:
                actions.append(RemediationAction.SCALE_UP)
            
            if metric == 'proxy_failure_rate' and severity == IncidentSeverity.CRITICAL:
                actions.extend([RemediationAction.ROTATE_PROXIES, RemediationAction.SWITCH_REGION])
            
            if metric == 'capture_success_rate' and severity == IncidentSeverity.CRITICAL:
                actions.extend([RemediationAction.ACTIVATE_STEALTH, RemediationAction.NOTIFY_ADMINS])
            
            if severity == IncidentSeverity.CRITICAL:
                actions.append(RemediationAction.NOTIFY_ADMINS)
        
        return list(set(actions))
    
    async def auto_remediate(self, incident: Incident):
        action_handlers = {
            RemediationAction.SCALE_UP: self.scale_up_resources,
            RemediationAction.SCALE_DOWN: self.scale_down_resources,
            RemediationAction.RESTART_PODS: self.restart_pods,
            RemediationAction.ROTATE_PROXIES: self.rotate_proxies,
            RemediationAction.SWITCH_REGION: self.switch_region,
            RemediationAction.ACTIVATE_STEALTH: self.activate_stealth_mode,
            RemediationAction.NOTIFY_ADMINS: self.notify_admins
        }
        
        for action in incident.remediation_actions:
            if action in action_handlers:
                try:
                    await action_handlers[action](incident)
                    logger.info(f"Executed remediation action: {action.name}")
                except Exception as e:
                    logger.error(f"Error executing action {action.name}: {e}")
    
    async def scale_up_resources(self, incident: Incident):
        pass
    
    async def rotate_proxies(self, incident: Incident):
        pass
    
    async def activate_stealth_mode(self, incident: Incident):
        pass
    
    async def notify_incident(self, incident: Incident):
        if not self.slack_client:
            return
        
        try:
            message = {
                "text": f"🚨 *System Incident Detected* 🚨",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "🚨 System Incident Detected"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*ID:* {incident.id}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Severity:* {incident.severity.name}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Detected:* {incident.detected_at}"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Description:* {incident.description}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Metrics:*\n" + "\n".join(
                                f"{k}: {v:.2f}" for k, v in incident.metrics.items()
                            )
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Remediation Actions:*\n" + "\n".join(
                                f"- {action.name}" for action in incident.remediation_actions
                            )
                        }
                    }
                ]
            }
            
            await self.slack_client.chat_postMessage(
                channel="#system-alerts",
                text=message["text"],
                blocks=message["blocks"]
            )
            
        except SlackApiError as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    def adjust_thresholds(self, metrics: Dict[str, float]):
        learning_rate = 0.1
        
        for metric_name, current_value in metrics.items():
            if metric_name in self.metric_baselines:
                baseline = self.metric_baselines[metric_name]
                
                baseline['mean'] = (learning_rate * current_value + 
                                  (1 - learning_rate) * baseline['mean'])
                
                if 'values_history' not in baseline:
                    baseline['values_history'] = []
                
                baseline['values_history'].append(current_value)
                if len(baseline['values_history']) > 100:
                    baseline['values_history'].pop(0)
                
                if len(baseline['values_history']) > 10:
                    baseline['std'] = np.std(baseline['values_history'])
                
                self.metric_thresholds[metric_name] = {
                    'warning': baseline['mean'] - baseline['std'],
                    'critical': baseline['mean'] - 2 * baseline['std']
                }
    
    def update_anomaly_detectors(self, metrics: Dict[str, float]):
        features = np.array([list(metrics.values())]).reshape(1, -1)
        
        self.anomaly_detector['isolation_forest'].fit(features)
    
    async def close(self):
        if self.slack_client:
            await self.slack_client.close()