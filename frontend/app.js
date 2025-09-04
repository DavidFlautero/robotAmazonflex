class AmazonFlexDashboard {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.isBotRunning = false;
        this.charts = {};
        this.initializeApp();
    }

    async initializeApp() {
        await this.checkAuthentication();
        await this.loadUserData();
        this.initializeCharts();
        this.setupEventListeners();
        this.startRealTimeUpdates();
    }

    async checkAuthentication() {
        try {
            const token = localStorage.getItem('auth_token');
            if (!token) {
                window.location.href = '/login';
                return;
            }

            const response = await fetch(`${this.apiBaseUrl}/api/user/stats`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                throw new Error('Not authenticated');
            }
        } catch (error) {
            console.error('Authentication error:', error);
            window.location.href = '/login';
        }
    }

    async loadUserData() {
        try {
            const token = localStorage.getItem('auth_token');
            const [statsResponse, userResponse] = await Promise.all([
                fetch(`${this.apiBaseUrl}/api/user/stats`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                }),
                fetch(`${this.apiBaseUrl}/api/user/profile`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                })
            ]);

            if (statsResponse.ok && userResponse.ok) {
                const stats = await statsResponse.json();
                const user = await userResponse.json();
                
                this.updateStatsDisplay(stats);
                this.updateUserInfo(user);
            }
        } catch (error) {
            console.error('Error loading user data:', error);
            this.addLogEntry('Error loading data. Please refresh the page.', 'error');
        }
    }

    updateStatsDisplay(stats) {
        document.getElementById('total-captures').textContent = stats.total_captures.toLocaleString();
        document.getElementById('success-rate').textContent = `${(stats.success_rate * 100).toFixed(1)}%`;
        document.getElementById('active-sessions').textContent = stats.active_sessions || '0';
        document.getElementById('monthly-earnings').textContent = `$${stats.monthly_earnings?.toLocaleString() || '0'}`;
    }

    updateUserInfo(user) {
        document.getElementById('user-email').textContent = user.email;
        
        if (user.settings) {
            document.getElementById('min-payment').value = user.settings.min_payment || 18;
            document.getElementById('max-distance').value = user.settings.max_distance || 20;
            document.getElementById('working-hours').value = user.settings.working_hours || '6-22';
        }
    }

    initializeCharts() {
        const successCtx = document.getElementById('successChart').getContext('2d');
        const earningsCtx = document.getElementById('earningsChart').getContext('2d');

        this.charts.successChart = new Chart(successCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                datasets: [{
                    label: 'Success Rate (%)',
                    data: Array(24).fill(0),
                    borderColor: '#00d4aa',
                    backgroundColor: 'rgba(0, 212, 170, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: '#e6e6e6' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#333' }
                    },
                    x: {
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#333' }
                    }
                }
            }
        });

        this.charts.earningsChart = new Chart(earningsCtx, {
            type: 'bar',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Daily Earnings ($)',
                    data: Array(7).fill(0),
                    backgroundColor: 'rgba(0, 212, 170, 0.8)',
                    borderColor: '#00d4aa',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { color: '#e6e6e6' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#333' }
                    },
                    x: {
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#333' }
                    }
                }
            }
        });
    }

    setupEventListeners() {
        document.getElementById('start-bot').addEventListener('click', () => this.startBot());
        document.getElementById('stop-bot').addEventListener('click', () => this.stopBot());
        document.getElementById('refresh-stats').addEventListener('click', () => this.refreshStats());
        document.getElementById('logout-btn').addEventListener('click', () => this.logout());

        // Settings change listeners
        document.getElementById('min-payment').addEventListener('change', (e) => this.updateSetting('min_payment', e.target.value));
        document.getElementById('max-distance').addEventListener('change', (e) => this.updateSetting('max_distance', e.target.value));
        document.getElementById('working-hours').addEventListener('change', (e) => this.updateSetting('working_hours', e.target.value));
    }

    async startBot() {
        if (this.isBotRunning) {
            this.addLogEntry('Bot is already running', 'warning');
            return;
        }

        try {
            const token = localStorage.getItem('auth_token');
            const response = await fetch(`${this.apiBaseUrl}/api/bot/start`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                this.isBotRunning = true;
                this.addLogEntry('Bot started successfully', 'success');
                document.getElementById('start-bot').disabled = true;
                document.getElementById('stop-bot').disabled = false;
            } else {
                throw new Error('Failed to start bot');
            }
        } catch (error) {
            console.error('Error starting bot:', error);
            this.addLogEntry('Error starting bot: ' + error.message, 'error');
        }
    }

    async stopBot() {
        if (!this.isBotRunning) {
            this.addLogEntry('Bot is not running', 'warning');
            return;
        }

        try {
            const token = localStorage.getItem('auth_token');
            const response = await fetch(`${this.apiBaseUrl}/api/bot/stop`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                this.isBotRunning = false;
                this.addLogEntry('Bot stopped successfully', 'info');
                document.getElementById('start-bot').disabled = false;
                document.getElementById('stop-bot').disabled = true;
            } else {
                throw new Error('Failed to stop bot');
            }
        } catch (error) {
            console.error('Error stopping bot:', error);
            this.addLogEntry('Error stopping bot: ' + error.message, 'error');
        }
    }

    async refreshStats() {
        await this.loadUserData();
        this.addLogEntry('Statistics refreshed', 'info');
    }

    async updateSetting(key, value) {
        try {
            const token = localStorage.getItem('auth_token');
            const response = await fetch(`${this.apiBaseUrl}/api/user/settings`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ [key]: value })
            });

            if (response.ok) {
                this.addLogEntry(`Setting updated: ${key} = ${value}`, 'success');
            }
        } catch (error) {
            console.error('Error updating setting:', error);
            this.addLogEntry('Error updating setting', 'error');
        }
    }

    addLogEntry(message, type = 'info') {
        const logContainer = document.getElementById('log-container');
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;

        // Keep only last 100 entries
        const entries = logContainer.querySelectorAll('.log-entry');
        if (entries.length > 100) {
            entries[0].remove();
        }
    }

    startRealTimeUpdates() {
        // Update charts every 30 seconds
        setInterval(() => this.updateCharts(), 30000);
        
        // Simulate real-time data for demo
        this.simulateRealTimeData();
    }

    async updateCharts() {
        try {
            const token = localStorage.getItem('auth_token');
            const response = await fetch(`${this.apiBaseUrl}/api/analytics/realtime`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (response.ok) {
                const data = await response.json();
                this.updateChartData(data);
            }
        } catch (error) {
            console.error('Error updating charts:', error);
        }
    }

    updateChartData(data) {
        if (data.success_rates) {
            this.charts.successChart.data.datasets[0].data = data.success_rates;
            this.charts.successChart.update();
        }

        if (data.daily_earnings) {
            this.charts.earningsChart.data.datasets[0].data = data.daily_earnings;
            this.charts.earningsChart.update();
        }
    }

    simulateRealTimeData() {
        // Simulate success rate data
        setInterval(() => {
            const newData = Array.from({length: 24}, () => Math.random() * 100);
            this.charts.successChart.data.datasets[0].data = newData;
            this.charts.successChart.update();
        }, 10000);

        // Simulate earnings data
        setInterval(() => {
            const newData = Array.from({length: 7}, () => Math.random() * 200);
            this.charts.earningsChart.data.datasets[0].data = newData;
            this.charts.earningsChart.update();
        }, 15000);

        // Simulate log entries
        const logMessages = [
            'Block captured: $25.50 - 3.2 miles',
            'Proxy rotated successfully',
            'New offer available: $28.75 - 2.1 miles',
            'Session refreshed',
            'Captcha detected, solving...',
            'Block accepted: $22.00 - 1.8 miles'
        ];

        setInterval(() => {
            const message = logMessages[Math.floor(Math.random() * logMessages.length)];
            const types = ['info', 'success', 'warning'];
            const type = types[Math.floor(Math.random() * types.length)];
            this.addLogEntry(message, type);
        }, 8000);
    }

    logout() {
        localStorage.removeItem('auth_token');
        window.location.href = '/login';
    }
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new AmazonFlexDashboard();
});