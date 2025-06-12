"""
TIER 1: QuantConnect API Client
High-performance async client for QuantConnect Cloud Platform integration
Based on Gemini's recommendations for async operations and <100ms response times
"""

import asyncio
import aiohttp
import time
import hashlib
import hmac
import base64
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import backoff

from tier1_core.logger import get_logger, SECURITY_LOGGER

@dataclass
class BacktestRequest:
    """Backtest request configuration"""
    project_id: int
    compile_id: str
    name: str
    note: str = ""
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class BacktestResult:
    """Backtest result data"""
    backtest_id: str
    project_id: int
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    charts: Optional[Dict[str, Any]] = None
    
    @property
    def is_complete(self) -> bool:
        return self.status == "Completed"
    
    @property
    def is_successful(self) -> bool:
        return self.is_complete and self.error is None
    
    @property
    def cagr(self) -> float:
        """Extract CAGR from statistics"""
        if self.statistics and "TotalPerformance" in self.statistics:
            return self.statistics["TotalPerformance"].get("CompoundAnnualReturn", 0.0)
        return 0.0
    
    @property
    def sharpe_ratio(self) -> float:
        """Extract Sharpe ratio from statistics"""
        if self.statistics and "TotalPerformance" in self.statistics:
            return self.statistics["TotalPerformance"].get("SharpeRatio", 0.0)
        return 0.0
    
    @property
    def max_drawdown(self) -> float:
        """Extract maximum drawdown from statistics"""
        if self.statistics and "TotalPerformance" in self.statistics:
            return abs(self.statistics["TotalPerformance"].get("Drawdown", 1.0))
        return 1.0

@dataclass
class ProjectInfo:
    """QuantConnect project information"""
    project_id: int
    name: str
    created: datetime
    modified: datetime
    language: str
    parameters: List[Dict[str, Any]]
    libraries: List[str]

class QuantConnectAPIError(Exception):
    """QuantConnect API specific error"""
    def __init__(self, message: str, status_code: int = None, response_data: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}

class QuantConnectClient:
    """
    High-performance async QuantConnect API client
    
    Features:
    - Async operations for <100ms response times
    - Automatic authentication with HMAC-SHA256
    - Rate limiting compliance (60 requests/minute)
    - Retry logic with exponential backoff
    - Connection pooling for performance
    - Comprehensive error handling
    """
    
    def __init__(self, user_id: str, token: str, api_url: str = "https://www.quantconnect.com/api/v2"):
        self.user_id = user_id
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.logger = get_logger(__name__)
        
        # Rate limiting
        self.request_times: List[float] = []
        self.max_requests_per_minute = 30  # More conservative limit
        self.backtest_request_times: List[float] = []
        self.max_backtests_per_minute = 2  # Limit to 2 concurrent backtests (2-node account)
        
        # Node management for 2-node account
        self.max_concurrent_backtests = 2
        self.active_backtests: List[str] = []
        self.backtest_queue: List[Dict[str, Any]] = []
        
        # Connection management
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector: Optional[aiohttp.TCPConnector] = None
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize the HTTP client session"""
        if self.session is None:
            # Configure connector for performance
            self.connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                enable_cleanup_closed=True
            )
            
            # Configure timeout
            timeout = aiohttp.ClientTimeout(
                total=30,  # Total timeout
                connect=10,  # Connection timeout
                sock_read=10  # Socket read timeout
            )
            
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers={
                    "User-Agent": "3-Tier-Evolution-System/1.0",
                    "Content-Type": "application/json"
                }
            )
            
        self.logger.info("QuantConnect client initialized")
    
    async def close(self) -> None:
        """Close the HTTP client session"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
        self.session = None
        self.connector = None
        self.logger.info("QuantConnect client closed")
    
    def _generate_auth_header(self, timestamp: str, method: str = "", path: str = "", body: str = "") -> str:
        """Generate QuantConnect API authentication header using correct format"""
        # QuantConnect API uses: API_TOKEN:timestamp -> SHA256 -> hex -> Base64(UserID:hashed_token)
        
        # Step 1: Create time-stamped token
        time_stamped_token = f"{self.token}:{timestamp}"
        
        # Step 2: Generate SHA256 hash (not HMAC)
        hashed_token = hashlib.sha256(time_stamped_token.encode('utf-8')).hexdigest()
        
        # Step 3: Create authentication string with UserID:hashed_token
        authentication = f"{self.user_id}:{hashed_token}"
        
        # Step 4: Base64 encode for Basic Auth
        auth_header = base64.b64encode(authentication.encode('utf-8')).decode('ascii')
        
        return auth_header
    
    async def _wait_for_rate_limit(self, endpoint: str = "") -> None:
        """Implement rate limiting to stay within API limits"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Special handling for backtest endpoints (more restrictive)
        if "backtest" in endpoint:
            self.backtest_request_times = [t for t in self.backtest_request_times if now - t < 60]
            
            if len(self.backtest_request_times) >= self.max_backtests_per_minute:
                sleep_time = 60 - (now - self.backtest_request_times[0]) + 2  # Extra buffer
                if sleep_time > 0:
                    self.logger.info(f"Backtest rate limiting: waiting {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            self.backtest_request_times.append(now)
        
        # General rate limiting
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                self.logger.debug(f"General rate limiting: waiting {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(now)
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, QuantConnectAPIError),
        max_tries=3,
        max_time=60,
        giveup=lambda e: "Too many" in str(e) and "slow down" in str(e)
    )
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated API request with retry logic"""
        start_time = time.time()
        
        try:
            # Ensure session is initialized
            if self.session is None:
                await self.initialize()
            
            # Rate limiting
            await self._wait_for_rate_limit(endpoint)
            
            # Prepare request
            url = f"{self.api_url}/{endpoint.lstrip('/')}"
            timestamp = str(int(datetime.now(timezone.utc).timestamp()))
            
            # Prepare body
            body = ""
            if data is not None:
                body = json.dumps(data, separators=(',', ':'))
            
            # Generate authentication
            auth_header = self._generate_auth_header(timestamp, method.upper(), f"/{endpoint.lstrip('/')}", body)
            
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Timestamp": timestamp
            }
            
            # Log the request (without sensitive data)
            SECURITY_LOGGER.log_api_call(endpoint, self.user_id, True)
            
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                data=body if data else None,
                params=params
            ) as response:
                
                # Update metrics
                response_time = time.time() - start_time
                self.total_requests += 1
                self.average_response_time = (
                    (self.average_response_time * (self.total_requests - 1) + response_time) / 
                    self.total_requests
                )
                
                # Check response
                response_data = await response.json()
                
                if response.status == 200 and response_data.get("success", False):
                    self.successful_requests += 1
                    
                    # Log performance if response time is concerning
                    if response_time > 0.1:  # 100ms target
                        self.logger.warning(f"Slow API response: {response_time:.3f}s for {endpoint}")
                    
                    return response_data
                else:
                    self.failed_requests += 1
                    error_msg = response_data.get("errors", [response_data.get("message", "Unknown error")])
                    
                    # Special handling for rate limiting
                    if any("too many" in str(err).lower() or "slow down" in str(err).lower() for err in error_msg):
                        self.logger.warning(f"Rate limit hit for {endpoint}, implementing backoff")
                        await asyncio.sleep(10)  # 10 second backoff for rate limiting
                    
                    raise QuantConnectAPIError(
                        f"API request failed: {error_msg}",
                        response.status,
                        response_data
                    )
                    
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Request failed for {endpoint}: {str(e)}")
            raise
    
    async def authenticate(self) -> bool:
        """Test authentication with QuantConnect"""
        try:
            response = await self._make_request("GET", "/authenticate")
            self.logger.info("✅ QuantConnect authentication successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ QuantConnect authentication failed: {str(e)}")
            SECURITY_LOGGER.log_auth_attempt(self.user_id, False, str(e))
            return False
    
    async def get_projects(self) -> List[ProjectInfo]:
        """Get list of user projects"""
        response = await self._make_request("GET", "/projects/read")
        
        projects = []
        for project_data in response.get("projects", []):
            projects.append(ProjectInfo(
                project_id=project_data["projectId"],
                name=project_data["name"],
                created=datetime.fromisoformat(project_data["created"].replace('Z', '+00:00')),
                modified=datetime.fromisoformat(project_data["modified"].replace('Z', '+00:00')),
                language=project_data["language"],
                parameters=project_data.get("parameters", []),
                libraries=project_data.get("libraries", [])
            ))
        
        return projects
    
    async def create_project(self, name: str, language: str = "Py") -> int:
        """Create a new project"""
        data = {
            "name": name,
            "language": language
        }
        
        response = await self._make_request("POST", "/projects/create", data)
        project_id = response["projects"][0]["projectId"]
        
        self.logger.info(f"Created project '{name}' with ID: {project_id}")
        return project_id
    
    async def update_project_files(self, project_id: int, files: Dict[str, str]) -> bool:
        """Update or create project files"""
        try:
            for filename, content in files.items():
                # Try to update existing file first
                try:
                    data = {
                        "projectId": project_id,
                        "name": filename,
                        "content": content
                    }
                    
                    # Try updating the file
                    response = await self._make_request("POST", "/files/update", data)
                    self.logger.debug(f"Updated file {filename} in project {project_id}")
                    
                except QuantConnectAPIError as e:
                    if "File extension is not valid" in str(e) or "file" in str(e).lower():
                        # If update fails, try creating the file instead
                        try:
                            create_data = {
                                "projectId": project_id,
                                "name": filename,
                                "content": content
                            }
                            
                            response = await self._make_request("POST", "/files/create", create_data)
                            self.logger.debug(f"Created file {filename} in project {project_id}")
                            
                        except Exception as create_error:
                            self.logger.error(f"Failed to create file {filename}: {str(create_error)}")
                            raise
                    else:
                        raise
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to update/create project files: {str(e)}")
            return False
    
    async def compile_project(self, project_id: int) -> str:
        """Compile a project and return compile ID"""
        data = {"projectId": project_id}
        
        response = await self._make_request("POST", "/compile/create", data)
        compile_id = response["compileId"]
        
        self.logger.info(f"Compiled project {project_id}, compile ID: {compile_id}")
        return compile_id
    
    async def create_backtest(self, backtest_request: BacktestRequest) -> str:
        """Create a new backtest with node management"""
        # Check if we can start immediately or need to queue
        if len(self.active_backtests) >= self.max_concurrent_backtests:
            # Queue the request
            self.backtest_queue.append({
                "request": backtest_request,
                "timestamp": time.time()
            })
            self.logger.info(f"Queuing backtest {backtest_request.name} - {len(self.active_backtests)}/{self.max_concurrent_backtests} nodes in use")
            
            # Wait for a slot to become available
            while len(self.active_backtests) >= self.max_concurrent_backtests:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self._check_backtest_completion()
        
        # Create the backtest
        data = {
            "projectId": backtest_request.project_id,
            "compileId": backtest_request.compile_id,
            "backtestName": backtest_request.name,
            "note": backtest_request.note,
            "parameters": backtest_request.parameters or {}
        }
        
        response = await self._make_request("POST", "/backtests/create", data)
        backtest_id = response["backtestId"]
        
        # Track the active backtest with project info
        self.active_backtests.append({
            "backtest_id": backtest_id,
            "project_id": backtest_request.project_id,
            "created_at": time.time(),
            "name": backtest_request.name
        })
        
        self.logger.info(f"Created backtest {backtest_id} for project {backtest_request.project_id} ({len(self.active_backtests)}/{self.max_concurrent_backtests} nodes)")
        return backtest_id
    
    async def _check_backtest_completion(self) -> None:
        """Check status of active backtests and remove completed ones"""
        if not self.active_backtests:
            return
        
        # Check each active backtest
        completed_backtests = []
        
        for backtest_info in self.active_backtests.copy():
            try:
                # Handle both old string format and new dict format for compatibility
                if isinstance(backtest_info, str):
                    # Old format - skip and remove after timeout
                    completed_backtests.append(backtest_info)
                    continue
                elif isinstance(backtest_info, dict):
                    # New format with complete info
                    result = await self.get_backtest_status(
                        backtest_info["project_id"], 
                        backtest_info["backtest_id"]
                    )
                    
                    # Check if completed or failed
                    if result.status in ["Completed", "RuntimeError", "Cancelled"]:
                        completed_backtests.append(backtest_info)
                        self.logger.info(f"Backtest {backtest_info['backtest_id']} completed with status: {result.status}")
                    
                    # Remove backtests that have been running too long (>2 hours)
                    elif time.time() - backtest_info["created_at"] > 7200:
                        completed_backtests.append(backtest_info)
                        self.logger.warning(f"Removing long-running backtest {backtest_info['backtest_id']} (timeout)")
                        
            except Exception as e:
                self.logger.error(f"Error checking backtest status for {backtest_info}: {str(e)}")
                # Remove problematic backtests to free up nodes
                completed_backtests.append(backtest_info)
        
        # Remove completed backtests
        for completed in completed_backtests:
            if completed in self.active_backtests:
                self.active_backtests.remove(completed)
        
        # Process queued backtests if nodes are available
        if len(self.active_backtests) < self.max_concurrent_backtests and self.backtest_queue:
            available_slots = self.max_concurrent_backtests - len(self.active_backtests)
            
            for _ in range(min(available_slots, len(self.backtest_queue))):
                if self.backtest_queue:
                    queued_item = self.backtest_queue.pop(0)
                    backtest_request = queued_item["request"]
                    
                    try:
                        # Create the queued backtest
                        data = {
                            "projectId": backtest_request.project_id,
                            "compileId": backtest_request.compile_id,
                            "backtestName": backtest_request.name,
                            "note": backtest_request.note,
                            "parameters": backtest_request.parameters or {}
                        }
                        
                        response = await self._make_request("POST", "/backtests/create", data)
                        backtest_id = response["backtestId"]
                        
                        # Track the new backtest
                        self.active_backtests.append({
                            "backtest_id": backtest_id,
                            "project_id": backtest_request.project_id,
                            "created_at": time.time(),
                            "name": backtest_request.name
                        })
                        
                        self.logger.info(f"Started queued backtest {backtest_id} - {len(self.active_backtests)}/{self.max_concurrent_backtests} nodes in use")
                        
                    except Exception as e:
                        self.logger.error(f"Error starting queued backtest: {str(e)}")
    
    async def cleanup_backtests(self) -> Dict[str, int]:
        """Periodic cleanup of backtests - returns status summary"""
        await self._check_backtest_completion()
        
        return {
            "active_backtests": len(self.active_backtests),
            "queued_backtests": len(self.backtest_queue),
            "max_concurrent": self.max_concurrent_backtests,
            "available_nodes": self.max_concurrent_backtests - len(self.active_backtests)
        }
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get current node utilization status"""
        return {
            "active_backtests": len(self.active_backtests),
            "queued_backtests": len(self.backtest_queue),
            "max_concurrent_backtests": self.max_concurrent_backtests,
            "available_nodes": self.max_concurrent_backtests - len(self.active_backtests),
            "node_utilization_percent": (len(self.active_backtests) / self.max_concurrent_backtests) * 100,
            "active_backtest_details": [
                {
                    "backtest_id": bt["backtest_id"], 
                    "name": bt["name"],
                    "runtime_minutes": (time.time() - bt["created_at"]) / 60
                } 
                for bt in self.active_backtests if isinstance(bt, dict)
            ]
        }
    
    async def get_backtest_status(self, project_id: int, backtest_id: str) -> BacktestResult:
        """Get backtest status and results"""
        params = {
            "projectId": project_id,
            "backtestId": backtest_id
        }
        
        response = await self._make_request("GET", "/backtests/read", params=params)
        backtest_data = response["backtests"][0]
        
        return BacktestResult(
            backtest_id=backtest_id,
            project_id=project_id,
            status=backtest_data["status"],
            progress=backtest_data.get("progress", 0.0),
            result=backtest_data.get("result"),
            error=backtest_data.get("error"),
            statistics=backtest_data.get("statistics"),
            charts=backtest_data.get("charts")
        )
    
    async def wait_for_backtest_completion(
        self, 
        project_id: int, 
        backtest_id: str, 
        timeout: int = 1800,  # 30 minutes
        poll_interval: int = 5
    ) -> BacktestResult:
        """Wait for backtest to complete with polling"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.get_backtest_status(project_id, backtest_id)
            
            if result.is_complete:
                self.logger.info(f"Backtest {backtest_id} completed in {time.time() - start_time:.1f}s")
                return result
            
            if result.status == "RuntimeError":
                raise QuantConnectAPIError(f"Backtest failed: {result.error}")
            
            await asyncio.sleep(poll_interval)
        
        raise QuantConnectAPIError(f"Backtest {backtest_id} timed out after {timeout}s")
    
    async def run_backtest_workflow(
        self, 
        project_id: int, 
        strategy_code: str, 
        strategy_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """Complete workflow: update code, compile, and run backtest"""
        try:
            # Update project files
            files = {"main.py": strategy_code}
            await self.update_project_files(project_id, files)
            
            # Compile project
            compile_id = await self.compile_project(project_id)
            
            # Create backtest
            backtest_request = BacktestRequest(
                project_id=project_id,
                compile_id=compile_id,
                name=strategy_name,
                note=f"Auto-generated strategy: {strategy_name}",
                parameters=parameters or {}
            )
            
            backtest_id = await self.create_backtest(backtest_request)
            
            # Wait for completion
            result = await self.wait_for_backtest_completion(project_id, backtest_id)
            
            self.logger.info(
                f"Backtest workflow complete: {strategy_name} - "
                f"CAGR: {result.cagr:.2%}, Sharpe: {result.sharpe_ratio:.2f}, "
                f"Drawdown: {result.max_drawdown:.2%}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest workflow failed for {strategy_name}: {str(e)}")
            raise
    
    async def batch_backtest(
        self, 
        project_id: int, 
        strategies: List[Dict[str, Any]], 
        max_concurrent: int = 5
    ) -> List[BacktestResult]:
        """Run multiple backtests concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_backtest(strategy_data: Dict[str, Any]) -> BacktestResult:
            async with semaphore:
                return await self.run_backtest_workflow(
                    project_id=project_id,
                    strategy_code=strategy_data["code"],
                    strategy_name=strategy_data["name"],
                    parameters=strategy_data.get("parameters")
                )
        
        tasks = [run_single_backtest(strategy) for strategy in strategies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch backtest failed for strategy {i}: {str(result)}")
            else:
                successful_results.append(result)
        
        self.logger.info(f"Batch backtest completed: {len(successful_results)}/{len(strategies)} successful")
        return successful_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        success_rate = (
            self.successful_requests / self.total_requests * 100 
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": success_rate,
            "average_response_time_ms": self.average_response_time * 1000,
            "rate_limit_compliant": len(self.request_times) < self.max_requests_per_minute
        }