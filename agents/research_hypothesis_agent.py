"""
Research Hypothesis Agent - Scrapes market news, analyzes content,
generates, and validates research hypotheses.
"""
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from agents.base_agent import BaseAgent, AgentConfig as BaseAgentConfig
from agents.knowledge_base import SharedKnowledgeBase, MarketInsight
# Assuming ResearchAgentSettings will be correctly defined in config.settings
# from config.settings import ResearchAgentSettings # Will be passed in __init__
from tier1_core.real_time_dashboard import DASHBOARD


# --- Helper Classes (Placeholders, as they are mocked in tests) ---
class FirecrawlClient:
    _mock_api_key_warned = False # Class variable to ensure warning is printed only once
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if not api_key and not FirecrawlClient._mock_api_key_warned:
            print("Warning: FirecrawlClient initialized without an API key.")
            FirecrawlClient._mock_api_key_warned = True


    async def scrape_market_news(self, query: str = "latest market news", max_pages: int = 1) -> List[Dict[str, Any]]:
        # This would normally make an API call
        # self.logger.debug(f"FirecrawlClient: Mock scraping for '{query}', max_pages={max_pages}")
        return []

class ReasoningModel:
    def __init__(self, provider: str = "openai", model_name: str = "gpt-4-turbo"):
        self.provider = provider
        self.model_name = model_name
        # self.logger.debug(f"ReasoningModel: Initialized with {provider} - {model_name}")

    async def analyze_research_data(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This would normally make an API call to an LLM
        # self.logger.debug(f"ReasoningModel: Mock analysis for {len(articles)} articles.")
        return {}

# --- Hypothesis Dataclass ---
@dataclass
class Hypothesis:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    theme: str = ""
    summary: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    status: str = "new"  # "new", "validating", "confirmed", "refuted", "stale"
    supporting_articles: List[Any] = field(default_factory=list) # Could be URLs or article objects/IDs
    generated_at: datetime = field(default_factory=datetime.now)
    last_validated_at: Optional[datetime] = None


class ResearchHypothesisAgent(BaseAgent):
    def __init__(
        self,
        config: Any, # Expected to be ResearchAgentSettings instance
        knowledge_base: SharedKnowledgeBase,
        strategy_generator: Optional[Any] = None,
        strategy_tester: Optional[Any] = None,
    ):
        base_config = BaseAgentConfig(
            name=getattr(config, 'name', "ResearchHypothesisAgent"),
            category=getattr(config, 'category', "research"),
            max_concurrent_tasks=getattr(config, 'max_concurrent_tasks', 2),
            generation_batch_size=getattr(config, 'generation_batch_size', 0),
            min_sharpe_threshold=getattr(config, 'min_sharpe_threshold', 0.0),
            min_cagr_threshold=getattr(config, 'min_cagr_threshold', 0.0),
            risk_tolerance=getattr(config, 'risk_tolerance', 0.5),
            exploration_rate=getattr(config, 'exploration_rate', 0.0),
            communication_frequency=getattr(config, 'communication_frequency', 300)
        )
        super().__init__(base_config, strategy_generator, strategy_tester, knowledge_base)

        self.config = config
        self.active_hypotheses: List[Hypothesis] = []
        
        self.firecrawl_client = FirecrawlClient(api_key=getattr(self.config, 'firecrawl_api_key', None))
        self.reasoning_model = ReasoningModel(
            provider=getattr(self.config, 'reasoning_model_provider', "openai"),
            model_name=getattr(self.config, 'reasoning_model_name', "gpt-4-turbo")
        )
        self.logger.info(f"{self.config.name} initialized. Max hypotheses: {self.config.max_active_hypotheses}, Interval: {self.config.research_interval_hours}h")

    async def _initialize_agent(self) -> None:
        self.logger.info(f"{self.config.name} specific initialization complete.")
        DASHBOARD.log_agent_activity(self.config.name, "Agent specific initialization complete", {})

    def _add_hypothesis(self, hypothesis_data: Dict[str, Any]):
        confidence = hypothesis_data.get("confidence", 0.0)
        theme = hypothesis_data.get("theme", "Untitled Hypothesis")

        if confidence < self.config.min_confidence_threshold:
            self.logger.debug(f"Skipping hypothesis '{theme}' due to low confidence: {confidence:.2f} < {self.config.min_confidence_threshold}")
            return

        new_hypothesis = Hypothesis(
            theme=theme,
            summary=hypothesis_data.get("summary", ""),
            confidence=confidence,
            supporting_articles=hypothesis_data.get("supporting_articles", []),
            status="new",
            generated_at=datetime.now()
        )

        if len(self.active_hypotheses) >= self.config.max_active_hypotheses:
            self.active_hypotheses.sort(key=lambda h: h.confidence)
            if new_hypothesis.confidence > self.active_hypotheses[0].confidence:
                removed = self.active_hypotheses.pop(0)
                self.logger.info(f"Max hypotheses. Removed '{removed.theme}' (conf: {removed.confidence:.2f}) to add '{new_hypothesis.theme}'.")
                self.active_hypotheses.append(new_hypothesis)
                self.active_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
            else:
                self.logger.info(f"Max hypotheses. New hypothesis '{new_hypothesis.theme}' (conf: {new_hypothesis.confidence:.2f}) not added (lowest: {self.active_hypotheses[0].confidence:.2f}).")
        else:
            self.active_hypotheses.append(new_hypothesis)
            self.active_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
            self.logger.info(f"Added new hypothesis: '{new_hypothesis.theme}' (conf: {new_hypothesis.confidence:.2f})")
        
        DASHBOARD.log_agent_activity(self.config.name, "Hypothesis added", {"theme": new_hypothesis.theme, "confidence": new_hypothesis.confidence})


    async def _conduct_research_cycle(self) -> None:
        self.logger.info(f"Starting new research cycle for {self.config.name}...")
        DASHBOARD.log_agent_activity(self.config.name, "Starting research cycle", {})

        query = "latest impactful financial news and market analysis"
        try:
            articles = await self.firecrawl_client.scrape_market_news(
                query=query,
                max_pages=self.config.firecrawl_max_pages
            )
            self.logger.info(f"Scraped {len(articles)} articles for query '{query}'.")
            DASHBOARD.log_agent_activity(self.config.name, "News scraped", {"article_count": len(articles)})
        except Exception as e:
            self.logger.error(f"Error during news scraping for {self.config.name}: {e}", exc_info=True)
            DASHBOARD.log_agent_activity(self.config.name, "News scraping error", {"error": str(e)})
            articles = []

        if not articles:
            self.logger.info("No articles found or error in scraping. Ending research cycle early.")
            return

        try:
            analysis_results = await self.reasoning_model.analyze_research_data(articles)
            self.logger.info(f"Research data analyzed by {self.config.name}. Found {len(analysis_results.get('key_themes', []))} key themes.")
            DASHBOARD.log_agent_activity(self.config.name, "Research analyzed", {"themes_found": len(analysis_results.get('key_themes', []))})
        except Exception as e:
            self.logger.error(f"Error during research data analysis for {self.config.name}: {e}", exc_info=True)
            DASHBOARD.log_agent_activity(self.config.name, "Research analysis error", {"error": str(e)})
            analysis_results = {}

        key_themes = analysis_results.get("key_themes", [])
        for theme_data in key_themes:
            self._add_hypothesis(theme_data)
            
            if theme_data.get("confidence", 0.0) >= self.config.min_confidence_threshold:
                insight = MarketInsight(
                    insight_id=f"{self.config.name}_hyp_{theme_data.get('theme', 'untitled').replace(' ', '_')}_{datetime.now().timestamp()}",
                    agent_name=self.config.name,
                    category="research_hypothesis",
                    asset_class="General",
                    symbols=[],
                    timeframe="General",
                    description=str(theme_data.get("summary", f"Hypothesis: {theme_data.get('theme')}"))[:250],
                    confidence=float(theme_data.get("confidence", 0.0)),
                    validity_period=timedelta(days=7),
                    supporting_data=theme_data,
                    timestamp=datetime.now()
                )
                await self.knowledge_base.add_market_insight(insight)
                self.logger.info(f"Added MarketInsight to KB for hypothesis: {theme_data.get('theme')}")

        self.logger.info(f"Research cycle complete for {self.config.name}.")

    async def _validate_hypotheses(self) -> None:
        self.logger.info(f"Starting validation for {len(self.active_hypotheses)} active hypotheses in {self.config.name}...")
        DASHBOARD.log_agent_activity(self.config.name, "Starting validation cycle", {"active_hypotheses": len(self.active_hypotheses)})
        
        valid_hypotheses = []
        hypothesis_stale_days = getattr(self.config, 'hypothesis_stale_days', 14) # Default if not in config

        for hypothesis in self.active_hypotheses:
            if datetime.now() - hypothesis.generated_at > timedelta(days=hypothesis_stale_days):
                hypothesis.status = "stale"
                self.logger.info(f"Hypothesis '{hypothesis.theme}' from {self.config.name} marked stale.")
                DASHBOARD.log_agent_activity(self.config.name, "Hypothesis stale", {"theme": hypothesis.theme, "id": hypothesis.id})
                continue

            if hypothesis.status == "new":
                hypothesis.status = "validating"
                hypothesis.last_validated_at = datetime.now()
                self.logger.info(f"Hypothesis '{hypothesis.theme}' status changed to validating for {self.config.name}.")
            
            valid_hypotheses.append(hypothesis)
        
        self.active_hypotheses = valid_hypotheses
        self.logger.info(f"Validation cycle complete for {self.config.name}.")

    async def run_cycle(self) -> None:
        self.state.current_task = "research_and_validation"
        self.state.status = "running_research_cycle"
        DASHBOARD.log_agent_activity(self.config.name, "Running full research/validation cycle", {})

        await self._conduct_research_cycle()
        await self._validate_hypotheses()

        self.state.current_task = "idle"
        self.state.status = "idle"
        DASHBOARD.log_agent_activity(self.config.name, "Full cycle finished, now idle", {})


    async def _main_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                await self.run_cycle()
                interval = self.config.research_interval_hours * 3600
                self.logger.info(f"{self.config.name} main loop complete, sleeping for {interval:.0f}s.")
                if interval <= 0:
                    self.logger.warning(f"Research interval is {interval}s. Agent will run once. Adjust config for periodic runs.")
                    break
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                self.logger.info(f"{self.config.name} main loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in {self.config.name} main loop: {e}", exc_info=True)
                self.state.last_error = str(e)
                DASHBOARD.log_agent_activity(self.config.name, "Main loop error", {"error": str(e)})
                await asyncio.sleep(300)

    async def _generate_strategies(self, insights: List[MarketInsight]) -> List[Any]:
        self.logger.info(f"{self.config.name} _generate_strategies called. Placeholder, returning empty list.")
        return []
