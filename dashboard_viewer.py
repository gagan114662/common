#!/usr/bin/env python3
"""
Real-Time Dashboard Viewer
Web interface for monitoring the 3-Tier Evolutionary Trading System
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Any
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socketserver

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tier1_core.real_time_dashboard import DASHBOARD
from tier1_core.logger import get_logger

logger = get_logger(__name__)

class DashboardHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for dashboard API"""
    
    def do_GET(self):
        if self.path == '/api/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get live dashboard data
            dashboard_data = DASHBOARD.get_live_dashboard_data()
            self.wfile.write(json.dumps(dashboard_data, indent=2, default=str).encode())
            
        elif self.path == '/api/agents':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get all agent details
            agents_data = {}
            for agent_name in DASHBOARD.active_agents.keys():
                agents_data[agent_name] = DASHBOARD.get_agent_details(agent_name)
            
            self.wfile.write(json.dumps(agents_data, indent=2, default=str).encode())
            
        elif self.path == '/api/events':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get recent events
            recent_events = [event.to_dict() for event in list(DASHBOARD.activity_events)[-50:]]
            self.wfile.write(json.dumps(recent_events, indent=2, default=str).encode())
            
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html_content = self.get_dashboard_html()
            self.wfile.write(html_content.encode())
            
        else:
            super().do_GET()
    
    def get_dashboard_html(self) -> str:
        """Generate the dashboard HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3-Tier Evolution System - Live Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .header {
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            text-align: center;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }
        .container { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 1rem; 
            padding: 1rem; 
            max-width: 1400px; 
            margin: 0 auto; 
        }
        .panel { 
            background: rgba(255,255,255,0.1); 
            border-radius: 10px; 
            padding: 1rem; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .panel h2 { 
            color: #ffd700; 
            margin-bottom: 1rem; 
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin: 0.5rem 0; 
            padding: 0.5rem;
            background: rgba(0,0,0,0.2);
            border-radius: 5px;
        }
        .metric-label { color: #ccc; }
        .metric-value { 
            font-weight: bold; 
            color: #00ff88;
        }
        .status-running { color: #00ff88; }
        .status-error { color: #ff4444; }
        .event-item {
            padding: 0.5rem;
            margin: 0.3rem 0;
            background: rgba(0,0,0,0.3);
            border-radius: 5px;
            border-left: 3px solid #ffd700;
            font-size: 0.9rem;
        }
        .event-time { color: #ccc; font-size: 0.8rem; }
        .event-source { color: #00ff88; font-weight: bold; }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #ffd700);
            transition: width 0.3s ease;
        }
        .auto-refresh {
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.5);
            padding: 0.5rem;
            border-radius: 5px;
            font-size: 0.8rem;
        }
        .grid-full { grid-column: 1 / -1; }
        .scroll-container { max-height: 300px; overflow-y: auto; }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.5); }
    </style>
</head>
<body>
    <div class="auto-refresh">🔄 Auto-refresh: <span id="refresh-countdown">5</span>s</div>
    
    <div class="header">
        <h1>🚀 3-Tier Evolutionary Trading System</h1>
        <p>Real-Time Strategy Development Dashboard</p>
        <p>Target: 25% CAGR | 1.0+ Sharpe | <15% Drawdown</p>
    </div>

    <div class="container">
        <!-- System Status -->
        <div class="panel">
            <h2>📊 System Status</h2>
            <div class="metric">
                <span class="metric-label">Status:</span>
                <span class="metric-value" id="system-status">Loading...</span>
            </div>
            <div class="metric">
                <span class="metric-label">Active Agents:</span>
                <span class="metric-value" id="active-agents">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Strategies Generated (1h):</span>
                <span class="metric-value" id="strategies-generated">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Strategies Tested (1h):</span>
                <span class="metric-value" id="strategies-tested">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Last Update:</span>
                <span class="metric-value" id="last-update">-</span>
            </div>
        </div>

        <!-- Performance Metrics -->
        <div class="panel">
            <h2>🎯 Performance Tracking</h2>
            <div class="metric">
                <span class="metric-label">Generation Rate:</span>
                <span class="metric-value" id="generation-rate">- /hour</span>
            </div>
            <div class="metric">
                <span class="metric-label">Test Success Rate:</span>
                <span class="metric-value" id="success-rate">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Avg Generation Time:</span>
                <span class="metric-value" id="avg-generation-time">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Avg Backtest Time:</span>
                <span class="metric-value" id="avg-backtest-time">-</span>
            </div>
        </div>

        <!-- Active Operations -->
        <div class="panel">
            <h2>🔄 Active Operations</h2>
            <div id="active-operations">
                <div class="metric">
                    <span class="metric-label">Active Generations:</span>
                    <span class="metric-value" id="active-generations">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Backtests:</span>
                    <span class="metric-value" id="active-backtests">-</span>
                </div>
                <div id="backtest-progress"></div>
            </div>
        </div>

        <!-- Agent Activity -->
        <div class="panel">
            <h2>🤖 Agent Activity</h2>
            <div id="agent-activity" class="scroll-container">
                Loading agent data...
            </div>
        </div>

        <!-- Recent Events -->
        <div class="panel grid-full">
            <h2>📋 Live Activity Feed</h2>
            <div id="recent-events" class="scroll-container">
                Loading events...
            </div>
        </div>
    </div>

    <script>
        let refreshInterval;
        let countdown = 5;

        function updateCountdown() {
            document.getElementById('refresh-countdown').textContent = countdown;
            countdown--;
            if (countdown < 0) {
                countdown = 5;
                fetchDashboardData();
            }
        }

        function formatTime(timeString) {
            const date = new Date(timeString);
            return date.toLocaleTimeString();
        }

        function formatDuration(seconds) {
            if (seconds < 60) return `${seconds.toFixed(1)}s`;
            return `${(seconds/60).toFixed(1)}m`;
        }

        async function fetchDashboardData() {
            try {
                // Fetch main dashboard data
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                
                // Update system status
                document.getElementById('system-status').textContent = data.system_status.toUpperCase();
                document.getElementById('system-status').className = `metric-value status-${data.system_status}`;
                document.getElementById('active-agents').textContent = Object.keys(data.active_agents).length;
                
                // Update performance summary
                const summary = data.performance_summary.last_hour;
                document.getElementById('strategies-generated').textContent = summary.strategies_generated;
                document.getElementById('strategies-tested').textContent = summary.strategies_tested;
                
                // Update metrics
                const metrics = data.system_metrics;
                document.getElementById('generation-rate').textContent = `${summary.generation_rate}/hour`;
                document.getElementById('success-rate').textContent = `${metrics.success_rate.toFixed(1)}%`;
                document.getElementById('avg-generation-time').textContent = formatDuration(metrics.average_generation_time);
                document.getElementById('avg-backtest-time').textContent = formatDuration(metrics.average_backtest_time);
                
                // Update active operations
                document.getElementById('active-generations').textContent = data.active_generations.length;
                document.getElementById('active-backtests').textContent = data.active_backtests.length;
                
                // Update timestamp
                document.getElementById('last-update').textContent = formatTime(data.timestamp);
                
                // Update agent activity
                updateAgentActivity(data.active_agents);
                
                // Fetch and update events
                await fetchRecentEvents();
                
            } catch (error) {
                console.error('Error fetching dashboard data:', error);
                document.getElementById('system-status').textContent = 'ERROR';
                document.getElementById('system-status').className = 'metric-value status-error';
            }
        }

        function updateAgentActivity(agents) {
            const container = document.getElementById('agent-activity');
            container.innerHTML = '';
            
            Object.entries(agents).forEach(([name, agent]) => {
                const agentDiv = document.createElement('div');
                agentDiv.className = 'event-item';
                agentDiv.innerHTML = `
                    <div class="event-source">${name}</div>
                    <div class="metric">
                        <span>Task:</span> 
                        <span>${agent.current_task || 'Idle'}</span>
                    </div>
                    <div class="metric">
                        <span>Success Rate:</span> 
                        <span>${agent.success_rate.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Completions:</span> 
                        <span>${agent.recent_completions}</span>
                    </div>
                `;
                container.appendChild(agentDiv);
            });
        }

        async function fetchRecentEvents() {
            try {
                const response = await fetch('/api/events');
                const events = await response.json();
                
                const container = document.getElementById('recent-events');
                container.innerHTML = '';
                
                events.slice(-20).reverse().forEach(event => {
                    const eventDiv = document.createElement('div');
                    eventDiv.className = 'event-item';
                    
                    const emoji = {
                        'strategy_generation': '🔄',
                        'backtesting': '🧪',
                        'evolution': '🧬',
                        'agent_action': '🤖'
                    }[event.event_type] || '📝';
                    
                    eventDiv.innerHTML = `
                        <div class="event-time">${formatTime(event.timestamp)}</div>
                        <div>${emoji} <span class="event-source">[${event.source}]</span> ${event.action}</div>
                        <div style="font-size: 0.8em; color: #ccc; margin-top: 0.2em;">
                            ${event.status} ${event.duration_ms ? `(${formatDuration(event.duration_ms/1000)})` : ''}
                        </div>
                    `;
                    container.appendChild(eventDiv);
                });
                
            } catch (error) {
                console.error('Error fetching events:', error);
            }
        }

        // Start the dashboard
        function startDashboard() {
            fetchDashboardData();
            refreshInterval = setInterval(updateCountdown, 1000);
        }

        // Initialize dashboard when page loads
        window.addEventListener('load', startDashboard);
    </script>
</body>
</html>
        '''

def start_dashboard_server(port=8080):
    """Start the dashboard web server"""
    try:
        with socketserver.TCPServer(("", port), DashboardHTTPHandler) as httpd:
            print(f"🌐 Dashboard server running at http://localhost:{port}")
            print("📊 Real-time monitoring available at:")
            print(f"   - Main Dashboard: http://localhost:{port}")
            print(f"   - API Endpoint: http://localhost:{port}/api/dashboard")
            print("🔄 Auto-refreshes every 5 seconds")
            print("\n💡 Press Ctrl+C to stop the server")
            
            # Try to open browser automatically
            try:
                webbrowser.open(f"http://localhost:{port}")
            except:
                pass
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🔒 Dashboard server stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard server: {e}")

def console_monitor():
    """Console-based real-time monitoring"""
    print("📊 Console Real-Time Monitor")
    print("=" * 50)
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H")
            
            # Header
            print("🚀 3-Tier Evolutionary Trading System - Live Monitor")
            print("=" * 60)
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Get dashboard data
            dashboard_data = DASHBOARD.get_live_dashboard_data()
            
            # System status
            print(f"📊 System Status: {dashboard_data['system_status'].upper()}")
            print(f"🤖 Active Agents: {len(dashboard_data['active_agents'])}")
            
            # Performance summary
            summary = dashboard_data['performance_summary']['last_hour']
            print(f"🔄 Generated (1h): {summary['strategies_generated']}")
            print(f"🧪 Tested (1h): {summary['strategies_tested']}")
            
            # Active operations
            print(f"⚡ Active Generations: {len(dashboard_data['active_generations'])}")
            print(f"🔬 Active Backtests: {len(dashboard_data['active_backtests'])}")
            
            print("\n🤖 Agent Activity:")
            print("-" * 30)
            for name, agent in dashboard_data['active_agents'].items():
                task = agent.get('current_task', 'Idle')
                success_rate = agent.get('success_rate', 0)
                print(f"  {name}: {task} (Success: {success_rate:.1f}%)")
            
            print("\n📋 Recent Events (last 5):")
            print("-" * 30)
            for event in dashboard_data['recent_events'][-5:]:
                time_str = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')).strftime('%H:%M:%S')
                emoji = {'strategy_generation': '🔄', 'backtesting': '🧪', 'evolution': '🧬', 'agent_action': '🤖'}.get(event['event_type'], '📝')
                print(f"  {time_str} {emoji} [{event['source']}] {event['action']} ({event['status']})")
            
            print("\n💡 Press Ctrl+C to exit")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🔒 Console monitor stopped")

def main():
    """Main function to choose monitoring method"""
    print("🚀 3-Tier Evolution System - Real-Time Dashboard")
    print("=" * 60)
    print("Choose your monitoring method:")
    print("1. 🌐 Web Dashboard (recommended)")
    print("2. 📺 Console Monitor")
    print("3. 📊 Both (web + console)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    # Start the dashboard
    DASHBOARD.start()
    print("✅ Dashboard started")
    
    try:
        if choice == "1":
            start_dashboard_server()
        elif choice == "2":
            console_monitor()
        elif choice == "3":
            # Start web server in a separate thread
            server_thread = threading.Thread(target=start_dashboard_server, daemon=True)
            server_thread.start()
            time.sleep(2)  # Give server time to start
            console_monitor()
        else:
            print("Invalid choice, starting web dashboard...")
            start_dashboard_server()
            
    except KeyboardInterrupt:
        print("\n🔒 Shutting down...")
    finally:
        DASHBOARD.stop()
        print("✅ Dashboard stopped")

if __name__ == "__main__":
    main()