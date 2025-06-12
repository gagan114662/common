"""
Geopolitical Risk Analysis Engine
Advanced modeling of geopolitical events and their market impact
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json
import re
import requests
from textblob import TextBlob
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """Geopolitical risk levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"

class EventType(Enum):
    """Types of geopolitical events"""
    MILITARY_CONFLICT = "military_conflict"
    TRADE_WAR = "trade_war"
    SANCTIONS = "sanctions"
    ELECTION = "election"
    POLICY_CHANGE = "policy_change"
    DIPLOMATIC_CRISIS = "diplomatic_crisis"
    TERRORISM = "terrorism"
    NATURAL_DISASTER = "natural_disaster"
    CYBER_ATTACK = "cyber_attack"
    CURRENCY_CRISIS = "currency_crisis"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"

class GeographicRegion(Enum):
    """Geographic regions for risk analysis"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    LATIN_AMERICA = "latin_america"
    GLOBAL = "global"

@dataclass
class GeopoliticalEvent:
    """Geopolitical event data structure"""
    event_id: str
    event_type: EventType
    description: str
    countries_involved: List[str]
    region: GeographicRegion
    severity_score: float
    probability: float
    market_impact_prediction: float
    affected_sectors: List[str]
    affected_currencies: List[str]
    start_date: datetime
    estimated_duration: Optional[timedelta]
    confidence_level: float
    data_sources: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskAssessment:
    """Risk assessment results"""
    overall_risk_level: RiskLevel
    risk_score: float
    contributing_events: List[str]
    sector_risks: Dict[str, float]
    currency_risks: Dict[str, float]
    geographic_risks: Dict[str, float]
    time_horizon_risks: Dict[str, float]
    risk_drivers: List[str]
    mitigation_strategies: List[str]
    assessment_timestamp: datetime

@dataclass
class MarketImpactModel:
    """Market impact prediction model"""
    model_id: str
    event_type: EventType
    historical_accuracy: float
    feature_importance: Dict[str, float]
    prediction_horizon: int  # days
    confidence_interval: Tuple[float, float]
    last_trained: datetime

class GeopoliticalRiskEngine:
    """
    Advanced geopolitical risk analysis and market impact prediction engine
    
    Features:
    - Real-time event monitoring
    - News sentiment analysis
    - Network analysis of country relationships
    - Machine learning impact prediction
    - Sector-specific risk assessment
    - Supply chain disruption modeling
    - Currency risk analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Event storage
        self.active_events: Dict[str, GeopoliticalEvent] = {}
        self.historical_events: List[GeopoliticalEvent] = []
        self.risk_assessments: List[RiskAssessment] = []
        
        # ML models
        self.impact_models: Dict[EventType, MarketImpactModel] = {}
        self.sentiment_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Network analysis
        self.country_network = nx.Graph()
        self.trade_relationships = {}
        self.alliance_matrix = {}
        
        # Risk parameters
        self.risk_thresholds = {
            RiskLevel.MINIMAL: 0.1,
            RiskLevel.LOW: 0.25,
            RiskLevel.MODERATE: 0.4,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 0.8,
            RiskLevel.EXTREME: 0.95
        }
        
        # Sector impact weights
        self.sector_weights = {
            'energy': 1.2,
            'defense': 1.1,
            'technology': 0.9,
            'financials': 1.0,
            'commodities': 1.3,
            'healthcare': 0.7,
            'consumer': 0.8,
            'industrials': 1.0
        }
        
        # Initialize components
        self._initialize_country_network()
        self._initialize_impact_models()
        
    def _initialize_country_network(self) -> None:
        """Initialize country relationship network"""
        
        # Major countries and their relationships
        countries = [
            'USA', 'CHN', 'RUS', 'GBR', 'DEU', 'FRA', 'JPN', 'IND', 'BRA', 'CAN',
            'AUS', 'KOR', 'TUR', 'IRN', 'SAU', 'ZAF', 'MEX', 'IDN', 'THA', 'SGP'
        ]
        
        # Add countries to network
        for country in countries:
            self.country_network.add_node(country)
        
        # Define trade relationships (simplified)
        trade_pairs = [
            ('USA', 'CHN', 0.8), ('USA', 'CAN', 0.9), ('USA', 'MEX', 0.7),
            ('CHN', 'JPN', 0.6), ('CHN', 'KOR', 0.7), ('CHN', 'DEU', 0.5),
            ('GBR', 'DEU', 0.8), ('GBR', 'FRA', 0.7), ('RUS', 'CHN', 0.6),
            ('RUS', 'DEU', 0.5), ('JPN', 'KOR', 0.6), ('IND', 'CHN', 0.4),
            ('DEU', 'FRA', 0.9), ('AUS', 'CHN', 0.7), ('SAU', 'USA', 0.6)
        ]
        
        # Add trade relationships as weighted edges
        for country1, country2, weight in trade_pairs:
            self.country_network.add_edge(country1, country2, weight=weight, type='trade')
        
        # Define alliance relationships
        alliance_pairs = [
            ('USA', 'GBR', 0.95), ('USA', 'CAN', 0.9), ('USA', 'AUS', 0.85),
            ('USA', 'JPN', 0.8), ('USA', 'KOR', 0.75), ('GBR', 'FRA', 0.8),
            ('GBR', 'DEU', 0.75), ('CHN', 'RUS', 0.6), ('FRA', 'DEU', 0.85)
        ]
        
        for country1, country2, strength in alliance_pairs:
            if self.country_network.has_edge(country1, country2):
                self.country_network[country1][country2]['alliance'] = strength
            else:
                self.country_network.add_edge(country1, country2, alliance=strength, type='alliance')
        
        self.logger.info(f"Initialized country network with {len(countries)} countries")
    
    def _initialize_impact_models(self) -> None:
        """Initialize market impact prediction models"""
        
        # Create simple models for each event type
        for event_type in EventType:
            model = MarketImpactModel(
                model_id=f"impact_model_{event_type.value}",
                event_type=event_type,
                historical_accuracy=0.65 + np.random.random() * 0.2,  # 65-85% accuracy
                feature_importance={
                    'severity': 0.3,
                    'countries_involved': 0.2,
                    'economic_ties': 0.15,
                    'market_sentiment': 0.2,
                    'duration': 0.15
                },
                prediction_horizon=30,  # 30 days
                confidence_interval=(0.1, 0.9),
                last_trained=datetime.now()
            )
            self.impact_models[event_type] = model
        
        self.logger.info(f"Initialized {len(self.impact_models)} market impact models")
    
    async def monitor_geopolitical_events(self, news_sources: List[str] = None) -> List[GeopoliticalEvent]:
        """Monitor and analyze geopolitical events from various sources"""
        
        if news_sources is None:
            news_sources = ['reuters', 'bloomberg', 'ap_news', 'bbc']
        
        detected_events = []
        
        # Simulate news monitoring (in production, would use real APIs)
        simulated_news = await self._simulate_news_feed()
        
        for news_item in simulated_news:
            # Analyze news for geopolitical events
            event = await self._analyze_news_item(news_item)
            
            if event:
                detected_events.append(event)
                
                # Store event
                self.active_events[event.event_id] = event
        
        self.logger.info(f"Detected {len(detected_events)} new geopolitical events")
        
        return detected_events
    
    async def _simulate_news_feed(self) -> List[Dict[str, Any]]:
        """Simulate news feed for demonstration"""
        
        # Simulate various types of geopolitical news
        simulated_news = [
            {
                'title': 'Trade tensions escalate between major economies',
                'content': 'New tariffs imposed affecting technology and automotive sectors',
                'source': 'reuters',
                'timestamp': datetime.now(),
                'countries': ['USA', 'CHN'],
                'sentiment': -0.6
            },
            {
                'title': 'Diplomatic meeting scheduled between allied nations',
                'content': 'Leaders to discuss defense cooperation and economic partnerships',
                'source': 'bloomberg',
                'timestamp': datetime.now(),
                'countries': ['USA', 'GBR', 'FRA'],
                'sentiment': 0.3
            },
            {
                'title': 'Supply chain disruptions reported in key manufacturing region',
                'content': 'Regional instability affecting global semiconductor production',
                'source': 'ap_news',
                'timestamp': datetime.now(),
                'countries': ['CHN', 'KOR', 'JPN'],
                'sentiment': -0.4
            },
            {
                'title': 'New sanctions package under consideration',
                'content': 'Economic measures targeting specific sectors and individuals',
                'source': 'bbc',
                'timestamp': datetime.now(),
                'countries': ['USA', 'RUS'],
                'sentiment': -0.7
            }
        ]
        
        return simulated_news
    
    async def _analyze_news_item(self, news_item: Dict[str, Any]) -> Optional[GeopoliticalEvent]:
        """Analyze individual news item for geopolitical events"""
        
        # Extract key information
        title = news_item.get('title', '')
        content = news_item.get('content', '')
        countries = news_item.get('countries', [])
        sentiment = news_item.get('sentiment', 0.0)
        
        # Classify event type
        event_type = self._classify_event_type(title + ' ' + content)
        
        if event_type is None:
            return None
        
        # Determine severity
        severity = self._calculate_severity(title, content, sentiment, countries)
        
        # Estimate probability
        probability = self._estimate_probability(event_type, severity, countries)
        
        # Predict market impact
        market_impact = await self._predict_market_impact(event_type, severity, countries)
        
        # Determine affected sectors and currencies
        affected_sectors = self._identify_affected_sectors(event_type, content)
        affected_currencies = self._identify_affected_currencies(countries)
        
        # Determine region
        region = self._determine_region(countries)
        
        # Create event
        event = GeopoliticalEvent(
            event_id=f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_events)}",
            event_type=event_type,
            description=f"{title}. {content}",
            countries_involved=countries,
            region=region,
            severity_score=severity,
            probability=probability,
            market_impact_prediction=market_impact,
            affected_sectors=affected_sectors,
            affected_currencies=affected_currencies,
            start_date=news_item.get('timestamp', datetime.now()),
            estimated_duration=self._estimate_duration(event_type, severity),
            confidence_level=0.7 + np.random.random() * 0.2,  # 70-90% confidence
            data_sources=[news_item.get('source', 'unknown')],
            metadata={
                'sentiment_score': sentiment,
                'original_title': title,
                'detection_method': 'news_analysis'
            }
        )
        
        return event
    
    def _classify_event_type(self, text: str) -> Optional[EventType]:
        """Classify geopolitical event type from text"""
        
        text_lower = text.lower()
        
        # Simple keyword-based classification
        if any(word in text_lower for word in ['war', 'conflict', 'military', 'invasion', 'attack']):
            return EventType.MILITARY_CONFLICT
        elif any(word in text_lower for word in ['trade', 'tariff', 'import', 'export', 'commerce']):
            return EventType.TRADE_WAR
        elif any(word in text_lower for word in ['sanction', 'embargo', 'restriction', 'ban']):
            return EventType.SANCTIONS
        elif any(word in text_lower for word in ['election', 'vote', 'campaign', 'president', 'prime minister']):
            return EventType.ELECTION
        elif any(word in text_lower for word in ['policy', 'regulation', 'law', 'legislation']):
            return EventType.POLICY_CHANGE
        elif any(word in text_lower for word in ['diplomatic', 'ambassador', 'embassy', 'foreign minister']):
            return EventType.DIPLOMATIC_CRISIS
        elif any(word in text_lower for word in ['terror', 'bomb', 'extremist', 'radical']):
            return EventType.TERRORISM
        elif any(word in text_lower for word in ['cyber', 'hack', 'malware', 'data breach']):
            return EventType.CYBER_ATTACK
        elif any(word in text_lower for word in ['currency', 'devaluation', 'forex', 'exchange rate']):
            return EventType.CURRENCY_CRISIS
        elif any(word in text_lower for word in ['supply chain', 'shortage', 'disruption', 'logistics']):
            return EventType.SUPPLY_CHAIN_DISRUPTION
        elif any(word in text_lower for word in ['earthquake', 'hurricane', 'flood', 'disaster']):
            return EventType.NATURAL_DISASTER
        
        return None
    
    def _calculate_severity(self, title: str, content: str, sentiment: float, countries: List[str]) -> float:
        """Calculate event severity score"""
        
        # Base severity from sentiment
        base_severity = max(0, -sentiment)  # Negative sentiment = higher severity
        
        # Adjust for country importance
        country_factor = 1.0
        major_powers = ['USA', 'CHN', 'RUS', 'GBR', 'DEU', 'FRA', 'JPN']
        
        for country in countries:
            if country in major_powers:
                country_factor *= 1.2
        
        # Adjust for keyword intensity
        intensity_keywords = ['crisis', 'emergency', 'urgent', 'critical', 'severe', 'major']
        text_lower = (title + ' ' + content).lower()
        intensity_factor = 1.0 + sum(0.1 for keyword in intensity_keywords if keyword in text_lower)
        
        # Calculate final severity
        severity = min(1.0, base_severity * country_factor * intensity_factor)
        
        return severity
    
    def _estimate_probability(self, event_type: EventType, severity: float, countries: List[str]) -> float:
        """Estimate probability of event occurrence/continuation"""
        
        # Base probabilities by event type
        base_probabilities = {
            EventType.MILITARY_CONFLICT: 0.3,
            EventType.TRADE_WAR: 0.6,
            EventType.SANCTIONS: 0.7,
            EventType.ELECTION: 0.9,
            EventType.POLICY_CHANGE: 0.8,
            EventType.DIPLOMATIC_CRISIS: 0.5,
            EventType.TERRORISM: 0.2,
            EventType.NATURAL_DISASTER: 0.4,
            EventType.CYBER_ATTACK: 0.4,
            EventType.CURRENCY_CRISIS: 0.3,
            EventType.SUPPLY_CHAIN_DISRUPTION: 0.6
        }
        
        base_prob = base_probabilities.get(event_type, 0.5)
        
        # Adjust for severity
        severity_adjustment = severity * 0.3
        
        # Adjust for country relationships
        relationship_adjustment = 0.0
        
        if len(countries) >= 2:
            for i, country1 in enumerate(countries):
                for country2 in countries[i+1:]:
                    if self.country_network.has_edge(country1, country2):
                        edge_data = self.country_network[country1][country2]
                        if 'alliance' in edge_data:
                            relationship_adjustment -= 0.1  # Allies less likely to conflict
                        elif edge_data.get('type') == 'trade' and event_type == EventType.TRADE_WAR:
                            relationship_adjustment += 0.2  # Trade partners more likely for trade disputes
        
        probability = min(0.95, max(0.05, base_prob + severity_adjustment + relationship_adjustment))
        
        return probability
    
    async def _predict_market_impact(self, event_type: EventType, severity: float, countries: List[str]) -> float:
        """Predict market impact using trained models"""
        
        if event_type not in self.impact_models:
            return severity * 0.5  # Default impact
        
        model = self.impact_models[event_type]
        
        # Create feature vector
        features = {
            'severity': severity,
            'countries_involved': len(countries),
            'economic_ties': self._calculate_economic_ties(countries),
            'market_sentiment': -0.3,  # Assume negative sentiment
            'duration': 30  # Default duration
        }
        
        # Simple weighted prediction
        prediction = 0.0
        for feature, weight in model.feature_importance.items():
            if feature in features:
                prediction += features[feature] * weight
        
        # Scale to [0, 1] range
        prediction = min(1.0, max(0.0, prediction))
        
        return prediction
    
    def _calculate_economic_ties(self, countries: List[str]) -> float:
        """Calculate economic ties strength between countries"""
        
        if len(countries) < 2:
            return 0.0
        
        total_ties = 0.0
        pair_count = 0
        
        for i, country1 in enumerate(countries):
            for country2 in countries[i+1:]:
                if self.country_network.has_edge(country1, country2):
                    edge_data = self.country_network[country1][country2]
                    trade_weight = edge_data.get('weight', 0.0)
                    total_ties += trade_weight
                    pair_count += 1
        
        return total_ties / pair_count if pair_count > 0 else 0.0
    
    def _identify_affected_sectors(self, event_type: EventType, content: str) -> List[str]:
        """Identify sectors likely to be affected by the event"""
        
        # Sector keywords mapping
        sector_keywords = {
            'energy': ['oil', 'gas', 'energy', 'petroleum', 'coal', 'renewable'],
            'defense': ['military', 'defense', 'weapon', 'security', 'army'],
            'technology': ['tech', 'software', 'semiconductor', 'ai', 'cyber', 'data'],
            'financials': ['bank', 'finance', 'insurance', 'credit', 'loan'],
            'commodities': ['commodity', 'metal', 'grain', 'agriculture', 'mining'],
            'healthcare': ['health', 'medical', 'pharma', 'drug', 'hospital'],
            'consumer': ['retail', 'consumer', 'goods', 'shopping', 'brand'],
            'industrials': ['manufacturing', 'industrial', 'factory', 'production']
        }
        
        # Default affected sectors by event type
        default_sectors = {
            EventType.MILITARY_CONFLICT: ['defense', 'energy', 'commodities'],
            EventType.TRADE_WAR: ['technology', 'industrials', 'consumer'],
            EventType.SANCTIONS: ['energy', 'financials', 'commodities'],
            EventType.CYBER_ATTACK: ['technology', 'financials'],
            EventType.SUPPLY_CHAIN_DISRUPTION: ['industrials', 'technology', 'consumer']
        }
        
        affected_sectors = set(default_sectors.get(event_type, []))
        
        # Add sectors based on content keywords
        content_lower = content.lower()
        for sector, keywords in sector_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                affected_sectors.add(sector)
        
        return list(affected_sectors)
    
    def _identify_affected_currencies(self, countries: List[str]) -> List[str]:
        """Identify currencies likely to be affected"""
        
        # Country to currency mapping (simplified)
        country_currency = {
            'USA': 'USD', 'CHN': 'CNY', 'RUS': 'RUB', 'GBR': 'GBP',
            'DEU': 'EUR', 'FRA': 'EUR', 'JPN': 'JPY', 'IND': 'INR',
            'BRA': 'BRL', 'CAN': 'CAD', 'AUS': 'AUD', 'KOR': 'KRW',
            'TUR': 'TRY', 'IRN': 'IRR', 'SAU': 'SAR', 'ZAF': 'ZAR',
            'MEX': 'MXN', 'IDN': 'IDR', 'THA': 'THB', 'SGP': 'SGD'
        }
        
        affected_currencies = set()
        
        for country in countries:
            if country in country_currency:
                affected_currencies.add(country_currency[country])
        
        # Always include USD as it's the global reserve currency
        affected_currencies.add('USD')
        
        return list(affected_currencies)
    
    def _determine_region(self, countries: List[str]) -> GeographicRegion:
        """Determine primary geographic region"""
        
        # Country to region mapping
        country_region = {
            'USA': GeographicRegion.NORTH_AMERICA,
            'CAN': GeographicRegion.NORTH_AMERICA,
            'MEX': GeographicRegion.NORTH_AMERICA,
            'GBR': GeographicRegion.EUROPE,
            'DEU': GeographicRegion.EUROPE,
            'FRA': GeographicRegion.EUROPE,
            'RUS': GeographicRegion.EUROPE,
            'TUR': GeographicRegion.EUROPE,
            'CHN': GeographicRegion.ASIA_PACIFIC,
            'JPN': GeographicRegion.ASIA_PACIFIC,
            'KOR': GeographicRegion.ASIA_PACIFIC,
            'IND': GeographicRegion.ASIA_PACIFIC,
            'AUS': GeographicRegion.ASIA_PACIFIC,
            'IDN': GeographicRegion.ASIA_PACIFIC,
            'THA': GeographicRegion.ASIA_PACIFIC,
            'SGP': GeographicRegion.ASIA_PACIFIC,
            'IRN': GeographicRegion.MIDDLE_EAST,
            'SAU': GeographicRegion.MIDDLE_EAST,
            'ZAF': GeographicRegion.AFRICA,
            'BRA': GeographicRegion.LATIN_AMERICA
        }
        
        if not countries:
            return GeographicRegion.GLOBAL
        
        # Count regions
        region_counts = {}
        for country in countries:
            region = country_region.get(country, GeographicRegion.GLOBAL)
            region_counts[region] = region_counts.get(region, 0) + 1
        
        # Return most common region
        if region_counts:
            return max(region_counts.items(), key=lambda x: x[1])[0]
        else:
            return GeographicRegion.GLOBAL
    
    def _estimate_duration(self, event_type: EventType, severity: float) -> Optional[timedelta]:
        """Estimate event duration"""
        
        # Base durations by event type (in days)
        base_durations = {
            EventType.MILITARY_CONFLICT: 180,
            EventType.TRADE_WAR: 365,
            EventType.SANCTIONS: 720,
            EventType.ELECTION: 30,
            EventType.POLICY_CHANGE: 90,
            EventType.DIPLOMATIC_CRISIS: 60,
            EventType.TERRORISM: 7,
            EventType.NATURAL_DISASTER: 14,
            EventType.CYBER_ATTACK: 3,
            EventType.CURRENCY_CRISIS: 30,
            EventType.SUPPLY_CHAIN_DISRUPTION: 45
        }
        
        base_days = base_durations.get(event_type, 30)
        
        # Adjust for severity
        duration_days = base_days * (0.5 + severity)
        
        return timedelta(days=int(duration_days))
    
    async def assess_overall_risk(self) -> RiskAssessment:
        """Assess overall geopolitical risk based on active events"""
        
        if not self.active_events:
            return RiskAssessment(
                overall_risk_level=RiskLevel.MINIMAL,
                risk_score=0.1,
                contributing_events=[],
                sector_risks={},
                currency_risks={},
                geographic_risks={},
                time_horizon_risks={},
                risk_drivers=[],
                mitigation_strategies=[],
                assessment_timestamp=datetime.now()
            )
        
        # Calculate weighted risk score
        total_risk = 0.0
        event_weights = 0.0
        contributing_events = []
        
        for event in self.active_events.values():
            # Weight by probability and severity
            weight = event.probability * event.severity_score
            total_risk += weight * event.market_impact_prediction
            event_weights += weight
            contributing_events.append(event.event_id)
        
        overall_risk_score = total_risk / event_weights if event_weights > 0 else 0.0
        
        # Determine risk level
        risk_level = RiskLevel.MINIMAL
        for level, threshold in self.risk_thresholds.items():
            if overall_risk_score >= threshold:
                risk_level = level
        
        # Calculate sector-specific risks
        sector_risks = self._calculate_sector_risks()
        
        # Calculate currency-specific risks
        currency_risks = self._calculate_currency_risks()
        
        # Calculate geographic risks
        geographic_risks = self._calculate_geographic_risks()
        
        # Calculate time horizon risks
        time_horizon_risks = self._calculate_time_horizon_risks()
        
        # Identify risk drivers
        risk_drivers = self._identify_risk_drivers()
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(risk_level, risk_drivers)
        
        assessment = RiskAssessment(
            overall_risk_level=risk_level,
            risk_score=overall_risk_score,
            contributing_events=contributing_events,
            sector_risks=sector_risks,
            currency_risks=currency_risks,
            geographic_risks=geographic_risks,
            time_horizon_risks=time_horizon_risks,
            risk_drivers=risk_drivers,
            mitigation_strategies=mitigation_strategies,
            assessment_timestamp=datetime.now()
        )
        
        # Store assessment
        self.risk_assessments.append(assessment)
        
        return assessment
    
    def _calculate_sector_risks(self) -> Dict[str, float]:
        """Calculate risk levels for different sectors"""
        
        sector_risks = {}
        
        for sector in self.sector_weights.keys():
            sector_risk = 0.0
            event_count = 0
            
            for event in self.active_events.values():
                if sector in event.affected_sectors:
                    impact = event.market_impact_prediction * event.probability
                    sector_risk += impact * self.sector_weights[sector]
                    event_count += 1
            
            if event_count > 0:
                sector_risk = sector_risk / event_count
                sector_risks[sector] = min(1.0, sector_risk)
        
        return sector_risks
    
    def _calculate_currency_risks(self) -> Dict[str, float]:
        """Calculate risk levels for different currencies"""
        
        currency_risks = {}
        currency_exposure = {}
        
        # Count currency exposure
        for event in self.active_events.values():
            for currency in event.affected_currencies:
                if currency not in currency_exposure:
                    currency_exposure[currency] = []
                currency_exposure[currency].append(event)
        
        # Calculate risk for each currency
        for currency, events in currency_exposure.items():
            total_risk = 0.0
            for event in events:
                risk = event.market_impact_prediction * event.probability * event.severity_score
                total_risk += risk
            
            currency_risks[currency] = min(1.0, total_risk / len(events))
        
        return currency_risks
    
    def _calculate_geographic_risks(self) -> Dict[str, float]:
        """Calculate risk levels for different geographic regions"""
        
        geographic_risks = {}
        region_events = {}
        
        # Group events by region
        for event in self.active_events.values():
            region = event.region.value
            if region not in region_events:
                region_events[region] = []
            region_events[region].append(event)
        
        # Calculate risk for each region
        for region, events in region_events.items():
            total_risk = 0.0
            for event in events:
                risk = event.market_impact_prediction * event.probability
                total_risk += risk
            
            geographic_risks[region] = min(1.0, total_risk / len(events))
        
        return geographic_risks
    
    def _calculate_time_horizon_risks(self) -> Dict[str, float]:
        """Calculate risks for different time horizons"""
        
        time_horizons = ['1_week', '1_month', '3_months', '1_year']
        time_horizon_risks = {}
        
        now = datetime.now()
        
        for horizon in time_horizons:
            if horizon == '1_week':
                cutoff = now + timedelta(weeks=1)
            elif horizon == '1_month':
                cutoff = now + timedelta(days=30)
            elif horizon == '3_months':
                cutoff = now + timedelta(days=90)
            else:  # 1_year
                cutoff = now + timedelta(days=365)
            
            relevant_events = []
            for event in self.active_events.values():
                if event.estimated_duration:
                    event_end = event.start_date + event.estimated_duration
                    if event_end <= cutoff:
                        relevant_events.append(event)
                else:
                    # Assume ongoing if no duration
                    relevant_events.append(event)
            
            if relevant_events:
                total_risk = sum(
                    event.market_impact_prediction * event.probability 
                    for event in relevant_events
                )
                time_horizon_risks[horizon] = min(1.0, total_risk / len(relevant_events))
            else:
                time_horizon_risks[horizon] = 0.0
        
        return time_horizon_risks
    
    def _identify_risk_drivers(self) -> List[str]:
        """Identify primary risk drivers"""
        
        risk_drivers = []
        
        # Count event types
        event_type_counts = {}
        for event in self.active_events.values():
            event_type = event.event_type.value
            impact = event.market_impact_prediction * event.probability
            
            if event_type not in event_type_counts:
                event_type_counts[event_type] = {'count': 0, 'total_impact': 0.0}
            
            event_type_counts[event_type]['count'] += 1
            event_type_counts[event_type]['total_impact'] += impact
        
        # Sort by impact and count
        sorted_types = sorted(
            event_type_counts.items(),
            key=lambda x: x[1]['total_impact'] + x[1]['count'] * 0.1,
            reverse=True
        )
        
        # Add top risk drivers
        for event_type, data in sorted_types[:5]:
            if data['total_impact'] > 0.1:
                risk_drivers.append(event_type.replace('_', ' ').title())
        
        return risk_drivers
    
    def _generate_mitigation_strategies(self, risk_level: RiskLevel, risk_drivers: List[str]) -> List[str]:
        """Generate risk mitigation strategies"""
        
        strategies = []
        
        # General strategies by risk level
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.EXTREME]:
            strategies.extend([
                "Reduce position sizes across all assets",
                "Increase cash allocation for liquidity",
                "Implement dynamic hedging strategies",
                "Monitor positions more frequently"
            ])
        
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.EXTREME]:
            strategies.extend([
                "Consider portfolio de-risking",
                "Activate circuit breakers",
                "Prepare for market volatility spikes"
            ])
        
        # Specific strategies by risk drivers
        for driver in risk_drivers:
            if 'trade' in driver.lower():
                strategies.append("Diversify across non-trade-sensitive sectors")
            elif 'military' in driver.lower() or 'conflict' in driver.lower():
                strategies.append("Increase allocation to defensive sectors")
            elif 'currency' in driver.lower():
                strategies.append("Implement currency hedging strategies")
            elif 'cyber' in driver.lower():
                strategies.append("Reduce exposure to technology-dependent assets")
            elif 'supply' in driver.lower():
                strategies.append("Avoid supply-chain-dependent sectors")
        
        return list(set(strategies))  # Remove duplicates
    
    async def get_event_impact_forecast(self, event_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Get detailed impact forecast for specific event"""
        
        if event_id not in self.active_events:
            return {'error': f'Event {event_id} not found'}
        
        event = self.active_events[event_id]
        
        # Generate daily impact forecast
        daily_impacts = []
        
        for day in range(forecast_days):
            forecast_date = datetime.now() + timedelta(days=day)
            
            # Decay impact over time
            time_decay = np.exp(-day / 30)  # 30-day half-life
            
            # Base impact
            base_impact = event.market_impact_prediction * event.probability
            
            # Daily impact with decay and random variation
            daily_impact = base_impact * time_decay * (0.8 + 0.4 * np.random.random())
            
            daily_impacts.append({
                'date': forecast_date.isoformat(),
                'impact': daily_impact,
                'confidence': event.confidence_level * time_decay
            })
        
        # Sector-specific impacts
        sector_impacts = {}
        for sector in event.affected_sectors:
            sector_weight = self.sector_weights.get(sector, 1.0)
            sector_impacts[sector] = event.market_impact_prediction * sector_weight
        
        return {
            'event_id': event_id,
            'event_type': event.event_type.value,
            'daily_forecast': daily_impacts,
            'sector_impacts': sector_impacts,
            'affected_currencies': event.affected_currencies,
            'peak_impact_day': np.argmax([d['impact'] for d in daily_impacts]),
            'total_forecasted_impact': sum(d['impact'] for d in daily_impacts),
            'forecast_generated': datetime.now().isoformat()
        }
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""
        
        # Latest risk assessment
        latest_assessment = self.risk_assessments[-1] if self.risk_assessments else None
        
        # Active events summary
        active_events_summary = []
        for event in self.active_events.values():
            active_events_summary.append({
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'severity': event.severity_score,
                'probability': event.probability,
                'impact': event.market_impact_prediction,
                'countries': event.countries_involved,
                'region': event.region.value
            })
        
        # Risk trends (simplified)
        risk_trend = []
        if len(self.risk_assessments) >= 2:
            for assessment in self.risk_assessments[-7:]:  # Last 7 assessments
                risk_trend.append({
                    'timestamp': assessment.assessment_timestamp.isoformat(),
                    'risk_score': assessment.risk_score,
                    'risk_level': assessment.overall_risk_level.value
                })
        
        return {
            'current_risk_level': latest_assessment.overall_risk_level.value if latest_assessment else 'minimal',
            'current_risk_score': latest_assessment.risk_score if latest_assessment else 0.0,
            'active_events_count': len(self.active_events),
            'active_events': active_events_summary,
            'sector_risks': latest_assessment.sector_risks if latest_assessment else {},
            'currency_risks': latest_assessment.currency_risks if latest_assessment else {},
            'geographic_risks': latest_assessment.geographic_risks if latest_assessment else {},
            'risk_drivers': latest_assessment.risk_drivers if latest_assessment else [],
            'mitigation_strategies': latest_assessment.mitigation_strategies if latest_assessment else [],
            'risk_trend': risk_trend,
            'last_updated': datetime.now().isoformat()
        }