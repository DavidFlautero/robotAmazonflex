import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from prometheus_client import Counter, Gauge, Histogram
import random
from enum import Enum
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PROXY_SUCCESS_RATE = Gauge('proxy_success_rate', 'Success rate of proxies')
PROXY_RESPONSE_TIME = Histogram('proxy_response_time_seconds', 'Response time of proxies')
PROXY_SWITCH_COUNT = Counter('proxy_switch_total', 'Total proxy switches')
PROXY_HEALTH_CHECK = Gauge('proxy_health_status', 'Proxy health status (1=healthy, 0=unhealthy)')

class ProxyType(Enum):
    RESIDENTIAL = 1
    DATACENTER = 2
    MOBILE = 3
    ISP = 4

class ProxyHealth(Enum):
    HEALTHY = 1
    DEGRADED = 2
    UNHEALTHY = 3
    DEAD = 4

@dataclass
class ProxyConfig:
    max_retries: int = 3
    timeout: int = 30
    health_check_interval: int = 300
    max_concurrent_requests: int = 100
    min_success_rate: float = 0.8
    max_response_time: int = 5000

@dataclass
class ProxyPerformance:
    success_rate: float
    response_time: float
    last_used: datetime
    health: ProxyHealth
    consecutive_failures: int

class AdvancedAIProxyManager:
    def __init__(self, config: ProxyConfig = None):
        self.config = config or ProxyConfig()
        self.proxy_pool = {}
        self.performance_metrics = {}
        self.session = None
        self.rl_model = self.initialize_rl_model()
        self.optimizer = optim.Adam(self.rl_model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.health_check_task = None
        self.initialize_session()
        
    def initialize_rl_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def initialize_session(self):
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests,
            verify_ssl=False,
            force_close=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            trust_env=True
        )
    
    async def get_optimal_proxy(self, target_url: str, 
                              user_context: Dict) -> Optional[Dict]:
        try:
            features = self.extract_features(target_url, user_context)
            
            proxy_scores = {}
            for proxy_id, proxy_data in self.proxy_pool.items():
                if self.is_proxy_available(proxy_id):
                    proxy_features = self.get_proxy_features(proxy_id, features)
                    score = self.rl_model(torch.FloatTensor(proxy_features))
                    proxy_scores[proxy_id] = score.item()
            
            if not proxy_scores:
                logger.warning("No available proxies in pool")
                return None
            
            best_proxy_id = max(proxy_scores.items(), key=lambda x: x[1])[0]
            best_proxy = self.proxy_pool[best_proxy_id]
            
            PROXY_SWITCH_COUNT.inc()
            self.performance_metrics[best_proxy_id].last_used = datetime.utcnow()
            
            return best_proxy
            
        except Exception as e:
            logger.error(f"Error selecting optimal proxy: {e}")
            return None
    
    def extract_features(self, target_url: str, user_context: Dict) -> List[float]:
        features = [
            len(target_url),
            int('amazon' in target_url.lower()),
            int('flex' in target_url.lower()),
            int('api' in target_url.lower()),
            user_context.get('priority', 1),
            user_context.get('required_stealth_level', 1),
            user_context.get('session_duration', 0),
            user_context.get('previous_attempts', 0),
            datetime.utcnow().hour / 24,
            datetime.utcnow().weekday() / 7,
            self.get_geo_distance(user_context.get('location', 'US')),
            self.get_avg_success_rate(),
            self.get_avg_response_time(),
            self.get_current_load(),
            random.random()
        ]
        return features
    
    def get_proxy_features(self, proxy_id: str, base_features: List[float]) -> List[float]:
        proxy_data = self.proxy_pool[proxy_id]
        perf_data = self.performance_metrics.get(proxy_id, ProxyPerformance(0, 0, datetime.utcnow(), ProxyHealth.HEALTHY, 0))
        
        proxy_specific_features = [
            self.proxy_type_to_num(proxy_data.get('type', ProxyType.DATACENTER)),
            perf_data.success_rate,
            perf_data.response_time / 1000,
            (datetime.utcnow() - perf_data.last_used).total_seconds() / 3600,
            perf_data.consecutive_failures,
            int(perf_data.health == ProxyHealth.HEALTHY),
            random.random()
        ]
        
        return base_features + proxy_specific_features
    
    async def execute_request(self, proxy: Dict, method: str, url: str, 
                            **kwargs) -> Optional[aiohttp.ClientResponse]:
        for attempt in range(self.config.max_retries):
            try:
                start_time = datetime.utcnow()
                
                async with self.session.request(
                    method, url, proxy=proxy['url'], **kwargs
                ) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    self.update_proxy_performance(proxy['id'], True, response_time)
                    
                    if await self.is_valid_response(response):
                        return response
                    else:
                        raise Exception("Invalid response content")
                        
            except Exception as e:
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.update_proxy_performance(proxy['id'], False, response_time)
                
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
                    return None
                
                await asyncio.sleep(2 ** attempt)
    
    async def is_valid_response(self, response: aiohttp.ClientResponse) -> bool:
        try:
            if response.status not in [200, 201, 202, 204]:
                return False
            
            content = await response.text()
            if any(block_indicator in content for block_indicator in 
                  ['captcha', 'access denied', 'bot detected', 'security check']):
                return False
            
            headers = response.headers
            if 'x-amz-bot-detection' in headers or 'cf-chl-bypass' in headers:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return False
    
    def update_proxy_performance(self, proxy_id: str, success: bool, response_time: float):
        if proxy_id not in self.performance_metrics:
            self.performance_metrics[proxy_id] = ProxyPerformance(
                success_rate=1.0 if success else 0.0,
                response_time=response_time,
                last_used=datetime.utcnow(),
                health=ProxyHealth.HEALTHY,
                consecutive_failures=0
            )
        
        perf = self.performance_metrics[proxy_id]
        
        alpha = 0.1
        perf.success_rate = (alpha * (1.0 if success else 0.0) + 
                           (1 - alpha) * perf.success_rate)
        
        perf.response_time = (alpha * response_time + 
                            (1 - alpha) * perf.response_time)
        
        if success:
            perf.consecutive_failures = 0
        else:
            perf.consecutive_failures += 1
        
        perf.health = self.determine_health_status(perf)
        
        PROXY_SUCCESS_RATE.set(perf.success_rate)
        PROXY_RESPONSE_TIME.observe(perf.response_time / 1000)
        PROXY_HEALTH_CHECK.set(1 if perf.health == ProxyHealth.HEALTHY else 0)
        
        self.train_rl_model(proxy_id, success, response_time)
    
    def determine_health_status(self, perf: ProxyPerformance) -> ProxyHealth:
        if perf.success_rate < self.config.min_success_rate * 0.5:
            return ProxyHealth.DEAD
        elif perf.success_rate < self.config.min_success_rate:
            return ProxyHealth.UNHEALTHY
        elif (perf.response_time > self.config.max_response_time * 2 or 
              perf.consecutive_failures > 3):
            return ProxyHealth.DEGRADED
        else:
            return ProxyHealth.HEALTHY
    
    def train_rl_model(self, proxy_id: str, success: bool, response_time: float):
        try:
            reward = self.calculate_reward(success, response_time)
            
            proxy_data = self.proxy_pool[proxy_id]
            features = self.get_proxy_features(proxy_id, [0] * 15)
            
            current_prediction = self.rl_model(torch.FloatTensor(features))
            
            target = torch.FloatTensor([reward])
            loss = self.loss_fn(current_prediction, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
    
    def calculate_reward(self, success: bool, response_time: float) -> float:
        base_reward = 1.0 if success else -2.0
        
        time_penalty = min(response_time / self.config.max_response_time, 2.0)
        reward = base_reward - time_penalty
        
        return max(-3.0, min(3.0, reward))
    
    async def start_health_checks(self):
        self.health_check_task = asyncio.create_task(self.health_check_loop())
    
    async def health_check_loop(self):
        while True:
            try:
                await self.perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop failed: {e}")
                await asyncio.sleep(60)
    
    async def perform_health_checks(self):
        check_tasks = []
        for proxy_id in list(self.proxy_pool.keys()):
            check_tasks.append(self.check_proxy_health(proxy_id))
        
        await asyncio.gather(*check_tasks, return_exceptions=True)
    
    async def check_proxy_health(self, proxy_id: str):
        try:
            test_urls = [
                "https://www.amazon.com/",
                "https://httpbin.org/ip",
                "https://api.myip.com/"
            ]
            
            successes = 0
            total_time = 0
            
            for test_url in test_urls:
                start_time = datetime.utcnow()
                try:
                    async with self.session.get(
                        test_url, 
                        proxy=self.proxy_pool[proxy_id]['url'],
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            successes += 1
                            total_time += (datetime.utcnow() - start_time).total_seconds() * 1000
                except:
                    pass
            
            success_rate = successes / len(test_urls) if test_urls else 0
            avg_time = total_time / successes if successes else self.config.max_response_time * 2
            
            if proxy_id in self.performance_metrics:
                self.performance_metrics[proxy_id].success_rate = success_rate
                self.performance_metrics[proxy_id].response_time = avg_time
                
                if success_rate < 0.5:
                    self.performance_metrics[proxy_id].consecutive_failures += 1
            
        except Exception as e:
            logger.error(f"Health check failed for proxy $proxy_id: $e")
    
    def is_proxy_available(self, proxy_id: str) -> bool:
        if proxy_id not in self.performance_metrics:
            return True
        
        perf = self.performance_metrics[proxy_id]
        
        if perf.health in [ProxyHealth.UNHEALTHY, ProxyHealth.DEAD]:
            return False
        
        if perf.consecutive_failures > self.config.max_retries * 2:
            return False
        
        if (perf.health == ProxyHealth.DEGRADED and 
            (datetime.utcnow() - perf.last_used).total_seconds() < 300):
            return False
        
        return True
    
    def proxy_type_to_num(self, proxy_type: ProxyType) -> float:
        type_map = {
            ProxyType.RESIDENTIAL: 1.0,
            ProxyType.ISP: 0.8,
            ProxyType.MOBILE: 0.6,
            ProxyType.DATACENTER: 0.4
        }
        return type_map.get(proxy_type, 0.4)
    
    def get_geo_distance(self, location: str) -> float:
        return random.random()
    
    def get_avg_success_rate(self) -> float:
        if not self.performance_metrics:
            return 0.8
        return sum(p.success_rate for p in self.performance_metrics.values()) / len(self.performance_metrics)
    
    def get_avg_response_time(self) -> float:
        if not self.performance_metrics:
            return 1000
        return sum(p.response_time for p in self.performance_metrics.values()) / len(self.performance_metrics)
    
    def get_current_load(self) -> float:
        active_proxies = sum(1 for pid in self.proxy_pool if self.is_proxy_available(pid))
        total_proxies = len(self.proxy_pool)
        return active_proxies / total_proxies if total_proxies > 0 else 0
    
    async def close(self):
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.session:
            await self.session.close()