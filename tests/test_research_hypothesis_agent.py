import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.research_hypothesis_agent import ResearchHypothesisAgent, Hypothesis, FirecrawlClient, ReasoningModel
from agents.knowledge_base import SharedKnowledgeBase, MarketInsight
from config.settings import SYSTEM_CONFIG, AgentConfigs, ResearchAgentSettings # Ensure these are correctly importable
from tier2_strategy.strategy_generator import StrategyGenerator # For BaseAgent - Corrected Path
from tier2_strategy.strategy_tester import StrategyTester     # For BaseAgent - Corrected Path

class TestResearchHypothesisAgent(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.mock_knowledge_base = AsyncMock(spec=SharedKnowledgeBase)

        # Ensure ResearchAgentSettings can be instantiated correctly from SYSTEM_CONFIG or directly
        # If SYSTEM_CONFIG.agents.research_hypothesis is already a ResearchAgentSettings instance:
        # self.agent_config = SYSTEM_CONFIG.agents.research_hypothesis
        # Otherwise, instantiate directly for the test:
        self.agent_config = ResearchAgentSettings(
            name="TestResearchAgent",
            # version="1.0", # Part of settings, not directly used by agent's core logic unless explicitly coded
            # description="Test Research Agent", # Same as above
            research_interval_hours=1, # Using a small interval for testing if needed, though sleep is patched
            max_active_hypotheses=5,
            min_confidence_threshold=0.6,
            firecrawl_max_pages=3,
            firecrawl_api_key="test_firecrawl_key", # Add if client uses it, even if mocked
            reasoning_model_provider="test_provider",
            reasoning_model_name="test_model",
            category="research" # For BaseAgentConfig
        )

        self.mock_strategy_generator = MagicMock(spec=StrategyGenerator)
        self.mock_strategy_tester = MagicMock(spec=StrategyTester)

        # Patch external dependencies of ResearchHypothesisAgent
        self.firecrawl_client_patcher = patch('agents.research_hypothesis_agent.FirecrawlClient', spec_set=FirecrawlClient) # Use spec_set
        self.MockFirecrawlClient = self.firecrawl_client_patcher.start()
        self.mock_firecrawl_instance = self.MockFirecrawlClient.return_value
        self.mock_firecrawl_instance.scrape_market_news = AsyncMock(return_value=[])

        self.reasoning_model_patcher = patch('agents.research_hypothesis_agent.ReasoningModel', spec_set=ReasoningModel) # Use spec_set
        self.MockReasoningModel = self.reasoning_model_patcher.start()
        self.mock_reasoning_instance = self.MockReasoningModel.return_value
        self.mock_reasoning_instance.analyze_research_data = AsyncMock(return_value={})

        # Patch Dashboard
        self.base_dashboard_patcher = patch('agents.base_agent.DASHBOARD', new_callable=MagicMock)
        self.MockBaseDashboard = self.base_dashboard_patcher.start()

        self.research_dashboard_patcher = patch('agents.research_hypothesis_agent.DASHBOARD', new=self.MockBaseDashboard) # Use same mock
        self.MockResearchDashboard = self.research_dashboard_patcher.start()

        # Patch asyncio.sleep in the agent's module to speed up loops
        self.sleep_patcher = patch('agents.research_hypothesis_agent.asyncio.sleep', new=AsyncMock())
        self.sleep_patcher.start()

        self.agent = ResearchHypothesisAgent(
            knowledge_base=self.mock_knowledge_base,
            config=self.agent_config,
            strategy_generator=self.mock_strategy_generator,
            strategy_tester=self.mock_strategy_tester
        )
        # Agent state is set by BaseAgent. For tests not calling start(), manually set for clarity.
        self.agent.state.is_active = True
        self.agent.state.status = "idle" # Default after init for this agent.


    async def asyncTearDown(self):
        self.firecrawl_client_patcher.stop()
        self.reasoning_model_patcher.stop()
        self.base_dashboard_patcher.stop()
        self.research_dashboard_patcher.stop()
        self.sleep_patcher.stop()
        if hasattr(self.agent, 'state') and self.agent.state.is_active: # Check state before calling stop
            await self.agent.stop()

    async def test_agent_initialization(self):
        self.assertEqual(self.agent.knowledge_base, self.mock_knowledge_base)
        self.assertEqual(self.agent.config, self.agent_config)
        self.assertEqual(self.agent.config.name, "TestResearchAgent") # Check name from agent's config
        self.MockFirecrawlClient.assert_called_once_with(api_key=self.agent_config.firecrawl_api_key)
        self.MockReasoningModel.assert_called_once_with(provider=self.agent_config.reasoning_model_provider, model_name=self.agent_config.reasoning_model_name)
        self.assertFalse(self.agent.active_hypotheses) # Should start empty

    async def test_add_new_hypothesis(self):
        hypothesis_data = {"theme": "AI impact on tech stocks", "confidence": 0.7, "supporting_articles": ["url1"], "summary":"Summary"}
        self.agent._add_hypothesis(hypothesis_data)

        self.assertEqual(len(self.agent.active_hypotheses), 1)
        hypothesis = self.agent.active_hypotheses[0]
        self.assertIsInstance(hypothesis, Hypothesis)
        self.assertEqual(hypothesis.theme, "AI impact on tech stocks")
        self.assertEqual(hypothesis.confidence, 0.7)
        self.assertEqual(hypothesis.status, "new")

    async def test_add_hypothesis_obeys_max_limit(self):
        for i in range(self.agent.config.max_active_hypotheses):
            self.agent._add_hypothesis({"theme": f"Theme {i}", "confidence": 0.8 + i*0.01, "summary":f"S{i}"}) # Vary confidence

        # This new one has higher confidence than the lowest if max_active_hypotheses is e.g. 5 (0.8 vs 0.85)
        self.agent._add_hypothesis({"theme": "Overflow Theme", "confidence": 0.85, "summary":"SO"})

        self.assertEqual(len(self.agent.active_hypotheses), self.agent.config.max_active_hypotheses)
        # Check if the lowest confidence hypothesis was replaced (if agent implements this logic)
        lowest_confidence_after_add = min(h.confidence for h in self.agent.active_hypotheses)
        self.assertGreaterEqual(lowest_confidence_after_add, 0.81, "Lowest confidence hypothesis should have been replaced or this new one is among them")


    async def test_add_hypothesis_skips_low_confidence(self):
        hypothesis_data = {"theme": "Low confidence theme", "confidence": self.agent.config.min_confidence_threshold - 0.1, "summary":"S"}
        self.agent._add_hypothesis(hypothesis_data)
        self.assertEqual(len(self.agent.active_hypotheses), 0)

    async def test_conduct_research_cycle_no_new_data(self):
        await self.agent._conduct_research_cycle()

        self.mock_firecrawl_instance.scrape_market_news.assert_called_once()
        self.mock_reasoning_instance.analyze_research_data.assert_not_called() # No articles to analyze
        self.assertEqual(len(self.agent.active_hypotheses), 0)

    async def test_conduct_research_cycle_with_data_generates_hypothesis(self):
        mock_articles = [{"title": "AI News", "url": "url1", "content": "Content"}]
        self.mock_firecrawl_instance.scrape_market_news.return_value = mock_articles

        # Ensure theme_data has all fields Hypothesis dataclass expects from it, or handle defaults in _add_hypothesis
        theme_data_from_llm = {"theme": "AI boom", "confidence": 0.85, "supporting_articles": ["url1"], "summary": "AI is booming"}
        mock_analysis = {
            "market_sentiment": "positive",
            "confidence_level": 0.85, # Overall analysis confidence
            "key_themes": [theme_data_from_llm],
            "emerging_trends": [],
            "actionable_insights": ["Invest in AI"]
        }
        self.mock_reasoning_instance.analyze_research_data.return_value = mock_analysis

        await self.agent._conduct_research_cycle()

        self.mock_firecrawl_instance.scrape_market_news.assert_called_once()
        self.mock_reasoning_instance.analyze_research_data.assert_called_once_with(mock_articles)

        self.assertEqual(len(self.agent.active_hypotheses), 1)
        hypothesis = self.agent.active_hypotheses[0]
        self.assertEqual(hypothesis.theme, "AI boom")
        self.assertEqual(hypothesis.confidence, 0.85)
        self.mock_knowledge_base.add_market_insight.assert_called_once()

        insight_args = self.mock_knowledge_base.add_market_insight.call_args[0][0]
        self.assertIsInstance(insight_args, MarketInsight)
        self.assertEqual(insight_args.category, "research_hypothesis") # Test expects 'type' but MarketInsight has 'category'
        self.assertEqual(insight_args.supporting_data['theme'], "AI boom") # Test checks data['theme']


    async def test_validate_hypotheses_no_active_hypotheses(self):
        await self.agent._validate_hypotheses()
        # No assertions needed, just testing it runs without error and doesn't create hypotheses

    async def test_validate_hypotheses_updates_status(self):
        initial_hypothesis_data = {"theme": "Initial Theme", "confidence": 0.75, "supporting_articles": ["url1"], "summary":"S"}
        self.agent._add_hypothesis(initial_hypothesis_data)
        self.assertEqual(self.agent.active_hypotheses[0].status, "new")

        # Simulate time passing or some condition that makes it ready for validation
        # The current _validate_hypotheses changes "new" to "validating"
        await self.agent._validate_hypotheses()

        self.assertTrue(len(self.agent.active_hypotheses) > 0, "Hypothesis should still be active")
        updated_hypothesis = self.agent.active_hypotheses[0]
        self.assertEqual(updated_hypothesis.status, "validating")
        self.assertIsNotNone(updated_hypothesis.last_validated_at)


    async def test_run_cycle_calls_main_methods(self):
        self.agent._conduct_research_cycle = AsyncMock()
        self.agent._validate_hypotheses = AsyncMock()
        # self.agent._generate_strategies_from_hypotheses = AsyncMock() # Not implemented yet

        await self.agent.run_cycle()

        self.agent._conduct_research_cycle.assert_called_once()
        self.agent._validate_hypotheses.assert_called_once()


if __name__ == '__main__':
    unittest.main()
