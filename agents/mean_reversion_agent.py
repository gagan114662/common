"""
Mean Reversion Agent - Identifies potential mean reversion opportunities.
"""

import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Any

from agents.base_agent import BaseAgent, AgentConfig as BaseAgentConfig # Renamed to avoid clash
from agents.knowledge_base import SharedKnowledgeBase, MarketInsight
from tier1_core.real_time_dashboard import DASHBOARD
# Assuming MeanReversionAgentSettings will be in config.settings
# from config.settings import MeanReversionAgentSettings # Specific config type
from datetime import datetime, timedelta


class MeanReversionAgent(BaseAgent):
    
    def __init__(
        self,
        config: Any, # Specific config for this agent, e.g. MeanReversionAgentSettings instance
        knowledge_base: SharedKnowledgeBase,
        strategy_generator: Optional[Any] = None,
        strategy_tester: Optional[Any] = None,
    ):
        # config is expected to be an object with attributes like name, category, bollinger_window etc.
        # This will typically be an instance of MeanReversionAgentSettings from config.settings
        
        # Create the BaseAgent's specific AgentConfig
        base_config = BaseAgentConfig(
            name=getattr(config, 'name', "MeanReversionAgent"),
            category=getattr(config, 'category', "mean_reversion"),
            max_concurrent_tasks=getattr(config, 'max_concurrent_tasks', 3),
            generation_batch_size=getattr(config, 'generation_batch_size', 0),
            min_sharpe_threshold=getattr(config, 'min_sharpe_threshold', 0.0),
            min_cagr_threshold=getattr(config, 'min_cagr_threshold', 0.0),
            risk_tolerance=getattr(config, 'risk_tolerance', 0.5),
            exploration_rate=getattr(config, 'exploration_rate', 0.0),
            communication_frequency=getattr(config, 'communication_frequency', 60)
        )
        super().__init__(base_config, strategy_generator, strategy_tester, knowledge_base)
        
        # Store the detailed agent-specific config (which should be MeanReversionAgentSettings type)
        self.config = config
        self.symbols_watchlist: List[str] = self.config.initial_symbols_watchlist
        # Other parameters like bollinger_window, rsi_period are accessed via self.config.parameter_name

        # self.logger is initialized in BaseAgent's __init__ (called via super())
        # BaseAgent's self.config.name is the correct name to use for logging related to the agent instance.
        # In this class, self.config refers to MeanReversionAgentSettings.
        # BaseAgent's self.config (which is BaseAgentConfig type) holds the name used by its logger.
        # The name in MeanReversionAgentSettings is passed to BaseAgentConfig.
        # So, self.config.name here is correct.
        self.logger.info(f"{self.config.name} initialized with watchlist: {self.symbols_watchlist}")

    async def _initialize_agent(self) -> None:
        # self.config refers to the MeanReversionAgentSettings instance, which has a 'name' attribute.
        self.logger.info(f"{self.config.name} specific initialization complete.")
        DASHBOARD.log_agent_activity(
            self.config.name,
            "Agent specific initialization complete", # Matches test string
            {"initial_symbols": self.symbols_watchlist}
        )

    async def _fetch_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        # self.config is MeanReversionAgentSettings. Its 'name' attribute is correct for logging.
        # The BaseAgent's logger (self.logger) is already configured with this name via BaseAgentConfig.
        self.logger.debug(f"Fetching market data for {symbol} over period {self.config.data_fetch_period}")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=self.config.data_fetch_period)
            if data.empty:
                self.logger.warning(f"No data returned for {symbol}")
                DASHBOARD.log_agent_activity(self.config.name, f"No data for {symbol}", {"symbol": symbol})
                return None
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                self.logger.warning(f"Data for {symbol} missing one of {required_cols}.")
                DASHBOARD.log_agent_activity(self.config.name, f"Data for {symbol} missing columns", {"symbol": symbol})
                return None
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            DASHBOARD.log_agent_activity(self.config.name, f"Error fetching data for {symbol}: {str(e)}", {"symbol": symbol})
            return None

    def _calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        self.logger.debug(f"Calculating indicators for {symbol}")
        
        min_len_bollinger = self.config.bollinger_window
        # RSI calculation needs at least rsi_period for the rolling mean of gains/losses.
        # The diff() reduces length by 1. So, data needs to be at least rsi_period + 1.
        min_len_rsi = self.config.rsi_period + 1
        
        # For the test "test_calculate_indicators_insufficient_data_rsi",
        # it provides length = bollinger_window + rsi_period - 5.
        # The critical path is often the longest window required for any indicator.
        # Let's use a combined minimum length check based on what's needed for all calculations.
        # Bollinger bands need `bollinger_window` prices.
        # RSI needs `rsi_period` differences, so `rsi_period + 1` prices.
        # If data is shorter than the longest window required for an intermediate step, it can fail.
        # The test `test_calculate_indicators_insufficient_data_bollinger` checks for length < bollinger_window.
        # The test `test_calculate_indicators_insufficient_data_rsi` checks for length < bollinger_window + rsi_period (approx).
        # The most robust check is for the absolute minimum number of rows needed to produce *any* output for all indicators.
        # This is typically self.config.bollinger_window for BB and self.config.rsi_period + 1 for RSI.
        # If we want the *last row* to have all indicators, we need at least max(bb_window, rsi_period+1) rows.
        
        if data is None or len(data) < min_len_bollinger or len(data) < min_len_rsi:
            self.logger.info(f"Insufficient data for {symbol} to calculate all indicators. Data len: {len(data) if data is not None else 0}, Need BB: {min_len_bollinger}, Need RSI: {min_len_rsi}")
            return None # Test `test_calculate_indicators_insufficient_data_bollinger` expects None if < BB window

        df = data.copy()
        
        # Bollinger Bands
        df['SMA'] = df['Close'].rolling(window=self.config.bollinger_window, min_periods=self.config.bollinger_window).mean()
        df['StdDev'] = df['Close'].rolling(window=self.config.bollinger_window, min_periods=self.config.bollinger_window).std()
        df['UpperBand'] = df['SMA'] + (df['StdDev'] * self.config.bollinger_std_dev)
        df['LowerBand'] = df['SMA'] - (df['StdDev'] * self.config.bollinger_std_dev)

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).fillna(0.0)
        loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)

        # EMA for RSI is more common, but test implies SMA-like rolling mean for avg_gain/loss
        avg_gain = gain.rolling(window=self.config.rsi_period, min_periods=self.config.rsi_period).mean()
        avg_loss = loss.rolling(window=self.config.rsi_period, min_periods=self.config.rsi_period).mean()

        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
        df.loc[avg_loss == 0, 'RSI'] = 100.0
        # For test_calculate_indicators_insufficient_data_rsi, if avg_loss is NaN, rs is NaN, RSI is NaN.
        # The test `test_calculate_indicators_insufficient_data_rsi` passes a length that might still produce NaNs at the end for RSI.
        # `df['RSI'] = df['RSI'].fillna(50)` was in previous version, but tests might want to see NaNs if truly uncalculable.
        # The test `test_calculate_indicators_insufficient_data_rsi` asserts `assertTrue(data_with_indicators['RSI'].iloc[-1:].isnull().all())`

        # If after all calculations, the last row doesn't have key indicators, it might be problematic for _identify_opportunity
        # However, _identify_opportunity itself checks for NaNs in the last row.
        return df

    def _identify_opportunity(self, latest_data_row: pd.Series, symbol: str) -> Optional[Dict[str, Any]]:
        # latest_data_row is expected to be the last row from the DataFrame returned by _calculate_indicators
        self.logger.debug(f"Identifying MR opportunity for {symbol} using latest data from {latest_data_row.name if hasattr(latest_data_row, 'name') else 'index N/A'}")

        required_indicators = ['Close', 'LowerBand', 'UpperBand', 'RSI', 'Volume', 'SMA']
        # Check if all required indicators are present in the series
        if not all(ind in latest_data_row.index for ind in required_indicators):
            self.logger.debug(f"Skipping MR opportunity for {symbol}: One or more required indicators missing in latest_data_row. Available: {list(latest_data_row.index)}")
            return None
        # Check for NaN in critical indicators
        if latest_data_row[required_indicators].isnull().any():
            self.logger.debug(f"Skipping MR opportunity for {symbol}: NaN in critical indicators on {latest_data_row.name if hasattr(latest_data_row, 'name') else 'index N/A'}. Data: {latest_data_row[required_indicators].to_dict()}")
            return None

        if latest_data_row['Volume'] < self.config.min_volume_threshold:
            self.logger.debug(f"Skipping MR opportunity for {symbol}: Low volume {latest_data_row['Volume']} < {self.config.min_volume_threshold}.")
            return None

        score = 0.0
        opportunity_type = None
        indicators = {
            'close': float(latest_data_row['Close']), 'sma': float(latest_data_row['SMA']),
            'upper_band': float(latest_data_row['UpperBand']), 'lower_band': float(latest_data_row['LowerBand']),
            'rsi': float(latest_data_row['RSI']), 'volume': float(latest_data_row['Volume'])
        }

        close_price = latest_data_row['Close']
        lower_band = latest_data_row['LowerBand']
        upper_band = latest_data_row['UpperBand']
        rsi_val = latest_data_row['RSI'] # Renamed from rsi to avoid conflict with module
        sma = latest_data_row['SMA']

        # Buy opportunity
        if close_price < lower_band and rsi_val < self.config.rsi_oversold:
            opportunity_type = "buy_reversion"
            price_diff_factor = (lower_band - close_price) / (sma * 0.05 + 1e-9)
            rsi_factor = (self.config.rsi_oversold - rsi_val) / (self.config.rsi_oversold + 1e-9)
            score = np.clip((price_diff_factor + rsi_factor) / 2.0, 0.0, 1.0)

        # Sell opportunity
        elif close_price > upper_band and rsi_val > self.config.rsi_overbought:
            opportunity_type = "sell_reversion"
            price_diff_factor = (close_price - upper_band) / (sma * 0.05 + 1e-9)
            rsi_factor = (rsi_val - self.config.rsi_overbought) / ((100.0 - self.config.rsi_overbought) + 1e-9)
            score = np.clip((price_diff_factor + rsi_factor) / 2.0, 0.0, 1.0)

        if opportunity_type and score >= self.config.opportunity_score_threshold:
            self.logger.info(f"MR Opportunity: {opportunity_type} for {symbol} with score {score:.2f}")
            return {
                "symbol": symbol,
                "opportunity_type": opportunity_type,
                "score": float(score),
                "price": float(close_price),
                "indicators": indicators # This dict now contains float values
            }
        return None

    async def run_cycle(self):
        self.logger.info(f"{self.config.name} starting new MR analysis cycle.") # Use self.config.name
        DASHBOARD.log_agent_activity(self.config.name, "Starting MR analysis cycle", {"num_symbols": len(self.symbols_watchlist)}) # Use self.config.name

        opportunities_found = 0
        for symbol in self.symbols_watchlist:
            market_data_full = await self._fetch_market_data(symbol)
            if market_data_full is None:
                continue

            data_with_indicators = self._calculate_indicators(market_data_full, symbol)
            if data_with_indicators is None or data_with_indicators.empty:
                 self.logger.info(f"No valid indicators calculated for {symbol}.")
                 continue
            
            if data_with_indicators.iloc[-1].isnull().any(): # Check last row for critical NaNs
                self.logger.info(f"Last row of indicators for {symbol} contains NaN. Skipping opportunity identification.")
                continue
            
            latest_data_row = data_with_indicators.iloc[-1]
            opportunity = self._identify_opportunity(latest_data_row, symbol)

            if opportunity:
                opportunities_found += 1
                # Test expects insight.symbol and insight.data['opportunity_type']
                # Test also expects insight.type to be like 'mean_reversion_buy'
                # My previous agent code used category for this. Let's align.
                # The test checks: insight.symbol, insight.type, insight.data['opportunity_type']
                # MarketInsight has 'category', not 'type'. 'supporting_data' is 'data'.
                insight_category = f"mean_reversion_{opportunity['opportunity_type'].split('_')[0]}"

                insight = MarketInsight(
                    insight_id=f"{self.config.name}_{opportunity['opportunity_type']}_{symbol}_{datetime.now().timestamp()}", # Use self.config.name
                    agent_name=self.config.name, # Use self.config.name
                    category=insight_category,
                    asset_class="Equity",
                    symbols=[symbol], # Test checks insight.symbols == ["SYM1"]
                    timeframe=self.config.data_fetch_period,
                    description=f"Potential {opportunity['opportunity_type']} for {symbol} at {opportunity['price']:.2f}, score: {opportunity['score']:.2f}",
                    confidence=opportunity['score'],
                    validity_period=timedelta(hours=6),
                    supporting_data=opportunity,
                    timestamp=datetime.now()
                )
                await self.knowledge_base.add_market_insight(insight)
            
            await asyncio.sleep(0.01) # Shorter sleep, yfinance is usually okay

        self.logger.info(f"MR Analysis cycle complete. Opportunities found: {opportunities_found}")
        DASHBOARD.log_agent_activity(
            self.config.name, "MR Analysis cycle complete",  # Use self.config.name
            {"symbols_analyzed": len(self.symbols_watchlist), "opportunities_found": opportunities_found}
        )

    async def _main_loop(self) -> None:
        # This loop is started by BaseAgent.start()
        while not self.shutdown_event.is_set():
            try:
                self.state.current_task = "run_mr_cycle"
                self.state.status = "running_mr_cycle"
                await self.run_cycle()
                self.state.current_task = "idle"
                self.state.status = "idle"

                await asyncio.sleep(self.config.run_cycle_interval_seconds)
            except asyncio.CancelledError:
                self.logger.info(f"{self.config.name} main loop cancelled.") # Use self.config.name
                self.state.status = "cancelled"
                break
            except Exception as e:
                self.logger.error(f"Error in {self.config.name} main loop: {e}", exc_info=True) # Use self.config.name
                self.state.last_error = str(e)
                self.state.status = "error"
                DASHBOARD.log_agent_activity(self.config.name, "Main loop error", {"error": str(e)}) # Use self.config.name
                await asyncio.sleep(max(10, self.config.run_cycle_interval_seconds / 2))

    async def _generate_strategies(self, insights: List[MarketInsight]) -> List[Any]:
        self.logger.info(f"{self.config.name} _generate_strategies called, but not implemented for this agent version. Returning empty list.") # Use self.config.name
        return []
