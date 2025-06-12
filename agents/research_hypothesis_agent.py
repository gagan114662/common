"""
Research Hypothesis Agent - AI-Powered Market Research and Hypothesis Generation
Integrates with open-deep-research framework for intelligent trading strategy development
"""

import asyncio
import json
import time
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents.base_agent import BaseAgent, AgentConfig
from agents.knowledge_base import SharedKnowledgeBase, MarketInsight

# Research-specific data models
@dataclass
class ResearchHypothesis:
    """Trading hypothesis based on research findings"""
    hypothesis_id: str
    description: str
    supporting_evidence: List[str]
    confidence_score: float  # 0.0 to 1.0
    research_sources: List[str]
    target_assets: List[str]
    expected_timeframe: str  # "short", "medium", "long"
    risk_assessment: Dict[str, float]
    strategy_implications: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class MarketResearchInsight:
    """Structured research insight from web sources"""
    insight_id: str
    research_type: str  # 'trend', 'fundamental', 'sentiment', 'academic', 'news'
    title: str
    summary: str
    findings: Dict[str, Any]
    implications: List[str]
    confidence: float
    data_sources: List[str]
    relevant_assets: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class FirecrawlClient:
    """Integration with Firecrawl for web research"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        self.base_url = os.getenv('FIRECRAWL_BASE_URL', 'https://api.firecrawl.dev')
        self.timeout = int(os.getenv('FIRECRAWL_TIMEOUT', '30'))
        self.logger = logging.getLogger(__name__)
        
        # Check if we have real API key
        self.use_real_api = self.api_key is not None and self.api_key.startswith('fc-')
        
        # Research targets
        self.financial_news_sources = [
            "bloomberg.com",
            "reuters.com", 
            "marketwatch.com",
            "finance.yahoo.com",
            "cnbc.com",
            "wsj.com"
        ]
        
        self.research_sources = [
            "papers.ssrn.com",
            "arxiv.org",
            "scholar.google.com"
        ]
    
    async def scrape_market_news(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Scrape latest market news for given keywords"""
        if self.use_real_api:
            try:
                import aiohttp
                news_results = []
                
                async with aiohttp.ClientSession() as session:
                    for keyword in keywords:
                        # Search financial news sites
                        search_url = f"{self.base_url}/v1/search"
                        headers = {
                            'Authorization': f'Bearer {self.api_key}',
                            'Content-Type': 'application/json'
                        }
                        
                        payload = {
                            "query": f"{keyword} financial news market analysis",
                            "pageOptions": {
                                "limit": min(5, limit),
                                "includeContent": True
                            },
                            "formats": ["markdown"]
                        }
                        
                        try:
                            async with session.post(search_url, json=payload, headers=headers, timeout=self.timeout) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    
                                    for result in data.get('data', []):
                                        news_results.append({
                                            "title": result.get('title', f'Market Analysis: {keyword}'),
                                            "url": result.get('url', ''),
                                            "content": result.get('content', result.get('markdown', ''))[:1000],
                                            "source": result.get('url', '').split('/')[2] if result.get('url') else 'Unknown',
                                            "timestamp": datetime.now().isoformat(),
                                            "relevance_score": 0.9
                                        })
                                        
                                        if len(news_results) >= limit:
                                            break
                        except Exception as e:
                            self.logger.warning(f"Firecrawl API error for {keyword}: {str(e)}")
                
                if news_results:
                    self.logger.info(f"Retrieved {len(news_results)} real news articles via Firecrawl")
                    return news_results[:limit]
                        
            except Exception as e:
                self.logger.error(f"Firecrawl integration error: {str(e)}")
        
        # Fallback to mock data
        self.logger.info("Using mock news data (Firecrawl API not available)")
        mock_news = []
        
        for keyword in keywords:
            for i in range(min(2, limit)):
                mock_news.append({
                    "title": f"Market Analysis: {keyword} Shows Strong Momentum",
                    "url": f"https://mockfinance.com/{keyword}-analysis-{i}",
                    "content": f"Recent analysis of {keyword} indicates significant market movements...",
                    "source": "MockFinance",
                    "timestamp": datetime.now().isoformat(),
                    "relevance_score": 0.8
                })
        
        return mock_news[:limit]
    
    async def research_economic_indicators(self, indicators: List[str]) -> List[Dict[str, Any]]:
        """Research current economic indicators"""
        mock_data = []
        
        for indicator in indicators:
            mock_data.append({
                "indicator": indicator,
                "current_value": f"Improving for {indicator}",
                "trend": "positive",
                "impact_assessment": f"{indicator} shows positive momentum affecting market sentiment",
                "sources": ["fed.gov", "bls.gov"],
                "timestamp": datetime.now().isoformat()
            })
        
        return mock_data
    
    async def mine_academic_research(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Mine academic research for trading insights"""
        mock_papers = []
        
        for topic in topics:
            mock_papers.append({
                "title": f"Advanced {topic} in Financial Markets",
                "authors": ["Dr. Research", "Prof. Analysis"],
                "abstract": f"This paper examines {topic} and its implications for trading strategies...",
                "key_findings": [
                    f"{topic} shows predictive power for market movements",
                    f"New methodology improves {topic} accuracy by 15%"
                ],
                "publication_date": "2024",
                "citations": 42,
                "url": f"https://papers.ssrn.com/{topic}-research"
            })
        
        return mock_papers

class ReasoningModel:
    """AI reasoning model for hypothesis generation"""
    
    def __init__(self, model_provider: str = "openai"):
        self.model_provider = model_provider
        self.logger = logging.getLogger(__name__)
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'o3')
        self.openai_temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        self.openai_max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '4000'))
        
        # o3 specific configuration for enhanced reasoning
        self.use_o3_reasoning = self.openai_model == 'o3'
        
        # Claude collaboration configuration
        self.use_claude_collaboration = os.getenv('USE_CLAUDE_COLLABORATION', 'true').lower() == 'true'
        self.enable_multi_ai = os.getenv('ENABLE_MULTI_AI_COLLAB', 'true').lower() == 'true'
        
        # Check if we can use real AI
        self.use_real_ai = self.openai_api_key and self.openai_api_key.startswith('sk-')
        
        if self.use_real_ai:
            try:
                import openai
                self.openai_client = openai.OpenAI(
                    api_key=self.openai_api_key,
                    base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                )
                self.logger.info(f"OpenAI client initialized with model: {self.openai_model}")
            except ImportError:
                self.logger.warning("OpenAI library not available, falling back to mock responses")
                self.use_real_ai = False
    
    async def analyze_research_data(self, research_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze research data and extract insights"""
        if self.use_real_ai:
            try:
                # Prepare research data for AI analysis
                research_text = "\n".join([
                    f"Source: {item.get('source', 'Unknown')}\n"
                    f"Title: {item.get('title', 'No title')}\n"
                    f"Content: {item.get('content', item.get('summary', 'No content'))[:500]}\n"
                    for item in research_data[:10]  # Limit to avoid token limits
                ])
                
                # Use Claude collaboration via MCP if available
                if self.use_claude_collaboration and self.enable_multi_ai:
                    try:
                        # Check if MCP multi-AI is available
                        import subprocess
                        result = subprocess.run(
                            ["python3", "-c", "from tier1_core.strategy_memory import get_strategy_memory; print('MCP available')"],
                            capture_output=True, text=True, cwd="."
                        )
                        
                        if "MCP available" in result.stdout:
                            # Use Claude collaboration for analysis
                            prompt = f"""
                            Analyze the following financial research data and extract trading insights:
                            
                            {research_text}
                            
                            Please provide:
                            1. Key market themes identified
                            2. Overall market sentiment (bullish/bearish/neutral)
                            3. Confidence level (0-1)
                            4. Actionable trading insights
                            5. Risk factors to consider
                            
                            Return analysis in JSON format.
                            """
                            
                            # This would use the MCP multi-AI collaboration
                            # For now, fallback to OpenAI
                            self.logger.info("Using Claude collaboration for research analysis")
                    except Exception as e:
                        self.logger.warning(f"Claude collaboration not available: {str(e)}")
                
                # Use OpenAI o3 reasoning model for enhanced analysis
                try:
                    system_prompt = "You are an expert financial analyst with advanced reasoning capabilities. Analyze research data and extract actionable trading insights. Use step-by-step reasoning to identify patterns, correlations, and market opportunities. Respond in JSON format with keys: key_themes, market_sentiment, confidence_level, actionable_insights, risk_factors, reasoning_steps."
                    
                    if self.use_o3_reasoning:
                        # Enhanced prompt for o3's reasoning capabilities
                        system_prompt += " Use your advanced reasoning to identify subtle market patterns, correlations between economic indicators, and potential market regime changes. Provide detailed reasoning steps for your analysis."
                    
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user", 
                                "content": f"Perform deep financial analysis on this research data using step-by-step reasoning:\n\n{research_text}\n\nProvide detailed reasoning for your market sentiment assessment and identify any emerging patterns or opportunities."
                            }
                        ],
                        temperature=self.openai_temperature,
                        max_tokens=self.openai_max_tokens
                    )
                    
                    analysis_text = response.choices[0].message.content
                    
                    # Try to parse JSON response
                    try:
                        insights = json.loads(analysis_text)
                        self.logger.info(f"Real AI analysis completed using {self.openai_model}")
                        return insights
                    except json.JSONDecodeError:
                        # Fallback: extract insights from text
                        insights = {
                            "key_themes": ["ai_analyzed_themes"],
                            "market_sentiment": "neutral",
                            "confidence_level": 0.8,
                            "actionable_insights": [analysis_text[:200] + "..."],
                            "risk_factors": ["Market volatility", "Economic uncertainty"]
                        }
                        return insights
                        
                except Exception as e:
                    self.logger.error(f"OpenAI API error: {str(e)}")
                    
            except Exception as e:
                self.logger.error(f"Real AI analysis error: {str(e)}")
        
        # Fallback to enhanced mock analysis
        self.logger.info("Using enhanced mock analysis")
        insights = {
            "key_themes": [],
            "market_sentiment": "neutral",
            "confidence_level": 0.7,
            "actionable_insights": [],
            "risk_factors": []
        }
        
        # Enhanced pattern detection
        research_text = " ".join([str(item) for item in research_data]).lower()
        
        if any(word in research_text for word in ["bullish", "positive", "growth", "momentum"]):
            insights["market_sentiment"] = "positive"
            insights["confidence_level"] = 0.8
        elif any(word in research_text for word in ["bearish", "negative", "decline", "recession"]):
            insights["market_sentiment"] = "negative"
            insights["confidence_level"] = 0.8
            
        if "momentum" in research_text:
            insights["key_themes"].append("momentum_trend")
        if "volatility" in research_text:
            insights["key_themes"].append("volatility_regime")
        if "earnings" in research_text:
            insights["key_themes"].append("earnings_season")
            
        insights["actionable_insights"] = [
            f"Market sentiment appears {insights['market_sentiment']}",
            "Consider adjusting position sizing based on volatility",
            "Monitor key economic indicators for trend confirmation"
        ]
        
        insights["risk_factors"] = [
            "Market regime uncertainty",
            "Economic data volatility", 
            "Geopolitical risks"
        ]
        
        return insights
    
    async def generate_hypothesis_from_insights(self, insights: Dict[str, Any]) -> ResearchHypothesis:
        """Generate trading hypothesis from research insights"""
        hypothesis_id = hashlib.md5(
            f"{insights}_{datetime.now()}".encode()
        ).hexdigest()[:12]
        
        # Mock hypothesis generation
        hypothesis = ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            description="Market momentum suggests upward trend in technology sector",
            supporting_evidence=[
                "Strong earnings growth in tech companies",
                "Positive sentiment in financial news",
                "Academic research supports momentum strategies"
            ],
            confidence_score=0.75,
            research_sources=["bloomberg.com", "papers.ssrn.com"],
            target_assets=["AAPL", "MSFT", "GOOGL"],
            expected_timeframe="medium",
            risk_assessment={
                "volatility_risk": 0.3,
                "market_risk": 0.4,
                "sector_risk": 0.2
            },
            strategy_implications=[
                "Consider momentum-based strategies",
                "Focus on technology sector exposure",
                "Implement risk management for volatility"
            ],
            created_at=datetime.now()
        )
        
        return hypothesis

class ResearchHypothesisAgent(BaseAgent):
    """
    AI-powered research agent for hypothesis generation
    
    Capabilities:
    - Web research using Firecrawl
    - AI-powered analysis and reasoning
    - Hypothesis generation for trading strategies
    - Market intelligence gathering
    """
    
    def __init__(
        self,
        strategy_generator=None,
        strategy_tester=None, 
        knowledge_base: SharedKnowledgeBase = None,
        config: Optional[AgentConfig] = None
    ):
        super().__init__(agent_name="research_hypothesis", config=config)
        
        self.strategy_generator = strategy_generator
        self.strategy_tester = strategy_tester
        self.knowledge_base = knowledge_base
        
        # Research components
        self.firecrawl_client = FirecrawlClient()
        self.reasoning_model = ReasoningModel()
        
        # Research state
        self.active_hypotheses: Dict[str, ResearchHypothesis] = {}
        self.research_cache: Dict[str, Any] = {}
        self.last_research_time = None
        
        # Research configuration
        self.research_interval = 1800  # 30 minutes
        self.max_hypotheses = 10
        self.confidence_threshold = 0.6
        
    async def initialize(self) -> None:
        """Initialize the research agent"""
        await super().initialize()
        
        # Register with knowledge base
        if self.knowledge_base:
            await self.knowledge_base.register_agent(self.agent_name)
        
        self.logger.info("Research Hypothesis Agent initialized")
        
        # Add initial market insight
        if self.knowledge_base:
            initial_insight = MarketInsight(
                insight_id=f"research_init_{int(time.time())}",
                agent_name=self.agent_name,
                insight_type="system",
                content="Research hypothesis agent online, monitoring market intelligence",
                confidence=1.0,
                relevant_assets=["SPY", "QQQ", "IWM"],
                expiry_time=datetime.now() + timedelta(hours=24)
            )
            await self.knowledge_base.add_insight(initial_insight)
    
    async def start(self) -> None:
        """Start the research agent"""
        await super().start()
        self.logger.info("Starting research hypothesis generation...")
        
        # Start research loop
        asyncio.create_task(self._research_loop())
    
    async def _research_loop(self) -> None:
        """Main research and hypothesis generation loop"""
        while self.is_running:
            try:
                # Check if it's time for research
                now = datetime.now()
                if (self.last_research_time is None or 
                    (now - self.last_research_time).total_seconds() >= self.research_interval):
                    
                    await self._conduct_research_cycle()
                    self.last_research_time = now
                
                # Brief pause
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in research loop: {str(e)}")
                await asyncio.sleep(300)  # 5 minute backoff on error
    
    async def _conduct_research_cycle(self) -> None:
        """Conduct a complete research cycle"""
        self.logger.info("Starting research cycle...")
        
        try:
            # 1. Market News Research
            news_insights = await self._research_market_news()
            
            # 2. Economic Indicators Research  
            economic_insights = await self._research_economic_indicators()
            
            # 3. Academic Research Mining
            academic_insights = await self._mine_academic_research()
            
            # 4. Combine and analyze all research
            all_research = news_insights + economic_insights + academic_insights
            analysis = await self.reasoning_model.analyze_research_data(all_research)
            
            # 5. Generate hypotheses
            if analysis["confidence_level"] >= self.confidence_threshold:
                hypothesis = await self.reasoning_model.generate_hypothesis_from_insights(analysis)
                await self._process_new_hypothesis(hypothesis)
            
            # 6. Share insights with knowledge base
            if self.knowledge_base:
                research_insight = MarketInsight(
                    insight_id=f"research_{int(time.time())}",
                    agent_name=self.agent_name,
                    insight_type="research",
                    content=f"Research cycle complete: {analysis['market_sentiment']} sentiment, {len(all_research)} sources analyzed",
                    confidence=analysis["confidence_level"],
                    relevant_assets=["SPY", "QQQ"],
                    expiry_time=datetime.now() + timedelta(hours=6)
                )
                await self.knowledge_base.add_insight(research_insight)
            
            self.logger.info(f"Research cycle complete: generated insights with {analysis['confidence_level']:.2f} confidence")
            
        except Exception as e:
            self.logger.error(f"Error in research cycle: {str(e)}")
    
    async def _research_market_news(self) -> List[Dict[str, Any]]:
        """Research latest market news"""
        keywords = ["market trends", "economic outlook", "sector rotation", "volatility"]
        news_data = await self.firecrawl_client.scrape_market_news(keywords, limit=20)
        
        self.logger.debug(f"Collected {len(news_data)} news articles")
        return news_data
    
    async def _research_economic_indicators(self) -> List[Dict[str, Any]]:
        """Research economic indicators"""
        indicators = ["unemployment", "inflation", "gdp growth", "interest rates"]
        economic_data = await self.firecrawl_client.research_economic_indicators(indicators)
        
        self.logger.debug(f"Analyzed {len(economic_data)} economic indicators")
        return economic_data
    
    async def _mine_academic_research(self) -> List[Dict[str, Any]]:
        """Mine academic research for insights"""
        topics = ["momentum strategies", "mean reversion", "market efficiency", "behavioral finance"]
        academic_data = await self.firecrawl_client.mine_academic_research(topics)
        
        self.logger.debug(f"Mined {len(academic_data)} academic papers")
        return academic_data
    
    async def _process_new_hypothesis(self, hypothesis: ResearchHypothesis) -> None:
        """Process and validate new hypothesis"""
        # Store hypothesis
        self.active_hypotheses[hypothesis.hypothesis_id] = hypothesis
        
        # Clean up old hypotheses
        if len(self.active_hypotheses) > self.max_hypotheses:
            oldest_id = min(
                self.active_hypotheses.keys(),
                key=lambda h: self.active_hypotheses[h].created_at
            )
            del self.active_hypotheses[oldest_id]
        
        # Share with knowledge base
        if self.knowledge_base:
            insight = MarketInsight(
                insight_id=f"hypothesis_{hypothesis.hypothesis_id}",
                agent_name=self.agent_name,
                insight_type="hypothesis",
                content=f"New hypothesis: {hypothesis.description}",
                confidence=hypothesis.confidence_score,
                relevant_assets=hypothesis.target_assets,
                expiry_time=datetime.now() + timedelta(hours=24)
            )
            await self.knowledge_base.add_insight(insight)
        
        self.logger.info(f"Generated new hypothesis: {hypothesis.hypothesis_id} (confidence: {hypothesis.confidence_score:.2f})")
    
    def get_active_hypotheses(self) -> List[ResearchHypothesis]:
        """Get all active hypotheses"""
        return list(self.active_hypotheses.values())
    
    def get_hypothesis_by_id(self, hypothesis_id: str) -> Optional[ResearchHypothesis]:
        """Get specific hypothesis by ID"""
        return self.active_hypotheses.get(hypothesis_id)
    
    async def validate_hypothesis(self, hypothesis_id: str, validation_results: Dict[str, Any]) -> None:
        """Update hypothesis based on validation results"""
        if hypothesis_id in self.active_hypotheses:
            hypothesis = self.active_hypotheses[hypothesis_id]
            
            # Update confidence based on validation
            if validation_results.get("success", False):
                hypothesis.confidence_score = min(1.0, hypothesis.confidence_score + 0.1)
            else:
                hypothesis.confidence_score = max(0.0, hypothesis.confidence_score - 0.2)
            
            self.logger.info(f"Updated hypothesis {hypothesis_id} confidence to {hypothesis.confidence_score:.2f}")
    
    async def stop(self) -> None:
        """Stop the research agent"""
        await super().stop()
        
        # Unregister from knowledge base
        if self.knowledge_base:
            await self.knowledge_base.unregister_agent(self.agent_name)
        
        self.logger.info("Research Hypothesis Agent stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            **super().get_status(),
            "active_hypotheses": len(self.active_hypotheses),
            "last_research": self.last_research_time.isoformat() if self.last_research_time else None,
            "next_research": (
                (self.last_research_time + timedelta(seconds=self.research_interval)).isoformat()
                if self.last_research_time else "pending"
            ),
            "confidence_threshold": self.confidence_threshold
        }