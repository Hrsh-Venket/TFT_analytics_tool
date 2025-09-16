// TFT Analytics Frontend Application
// Main application logic for Firebase dashboard

class TFTAnalyticsApp {
    constructor() {
        this.currentTab = 'overview';
        this.clustersData = null;
        this.init();
    }

    init() {
        this.setupTabSwitching();
        this.setupQueryForm();
        this.loadInitialData();
        this.setupExampleQueries();
    }

    setupTabSwitching() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabId = button.id.replace('tab-', '');
                this.switchTab(tabId);
            });
        });
    }

    switchTab(tabId) {
        // Update button states
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.className = 'tab-button px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300';
        });
        document.getElementById(`tab-${tabId}`).className = 'tab-button px-4 py-2 bg-blue-500 text-white rounded';

        // Update content visibility
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        document.getElementById(`content-${tabId}`).classList.remove('hidden');

        this.currentTab = tabId;

        // Load tab-specific data
        if (tabId === 'clusters' && !this.clustersData) {
            this.loadClusters();
        } else if (tabId === 'logs') {
            this.loadFunctionStatus();
        }
    }

    async loadInitialData() {
        await this.loadStats();
    }

    async loadStats() {
        const container = document.getElementById('stats-container');
        const loading = document.getElementById('stats-loading');

        try {
            const stats = await window.tftAPI.getStats();
            const formatted = window.tftAPI.formatStats(stats);

            container.innerHTML = `
                <div class="bg-white p-6 rounded-lg shadow">
                    <h3 class="text-lg font-medium mb-2 text-gray-800">Total Matches</h3>
                    <p class="text-3xl font-bold text-blue-600">${formatted.matches}</p>
                </div>
                <div class="bg-white p-6 rounded-lg shadow">
                    <h3 class="text-lg font-medium mb-2 text-gray-800">Participants</h3>
                    <p class="text-3xl font-bold text-green-600">${formatted.participants}</p>
                    <p class="text-sm text-gray-500">Avg ${formatted.avgPlayersPerMatch}/match</p>
                </div>
                <div class="bg-white p-6 rounded-lg shadow">
                    <h3 class="text-lg font-medium mb-2 text-gray-800">Average Placement</h3>
                    <p class="text-3xl font-bold text-purple-600">${formatted.avgPlacement}</p>
                    <p class="text-sm text-gray-500">Level ${formatted.avgLevel}</p>
                </div>
            `;

            loading.classList.add('hidden');
            container.classList.remove('hidden');

        } catch (error) {
            console.error('Error loading stats:', error);
            container.innerHTML = `
                <div class="col-span-3 bg-red-50 border border-red-200 rounded-lg p-4">
                    <h3 class="text-red-800 font-medium">Connection Error</h3>
                    <p class="text-red-700 text-sm mt-1">Failed to load statistics: ${error.message}</p>
                    <p class="text-red-600 text-sm mt-2">Make sure Cloud Functions are deployed and PROJECT_ID is set in api.js</p>
                </div>
            `;
            loading.classList.add('hidden');
            container.classList.remove('hidden');
        }
    }

    async loadClusters() {
        const container = document.getElementById('clusters-container');
        const loading = document.getElementById('clusters-loading');

        try {
            const response = await window.tftAPI.getClusters();
            this.clustersData = response;

            if (!response.clusters || response.clusters.length === 0) {
                container.innerHTML = `
                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                        <h3 class="text-yellow-800 font-medium">No Clusters Available</h3>
                        <p class="text-yellow-700 text-sm mt-1">${response.message || 'Run clustering analysis to generate clusters.'}</p>
                    </div>
                `;
            } else {
                let html = '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">';

                response.clusters.forEach(cluster => {
                    html += `
                        <div class="bg-white p-4 rounded-lg shadow hover:shadow-md transition-shadow cursor-pointer"
                             onclick="app.showClusterDetails(${cluster.id})">
                            <h4 class="font-medium text-lg mb-2">${cluster.name}</h4>
                            <div class="space-y-1 text-sm text-gray-600">
                                <div><strong>Size:</strong> ${window.tftAPI.formatNumber(cluster.size)} compositions</div>
                                <div><strong>Avg Placement:</strong> ${cluster.avg_placement}</div>
                                <div><strong>Win Rate:</strong> ${cluster.winrate}%</div>
                                <div><strong>Top 4 Rate:</strong> ${cluster.top4_rate}%</div>
                            </div>
                            ${cluster.carries && cluster.carries.length > 0 ?
                                `<div class="mt-2"><strong>Carries:</strong> ${cluster.carries.join(', ')}</div>` :
                                ''
                            }
                        </div>
                    `;
                });

                html += '</div>';
                container.innerHTML = html;
            }

            loading.classList.add('hidden');
            container.classList.remove('hidden');

        } catch (error) {
            console.error('Error loading clusters:', error);
            container.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                    <h3 class="text-red-800 font-medium">Connection Error</h3>
                    <p class="text-red-700 text-sm mt-1">Failed to load clusters: ${error.message}</p>
                </div>
            `;
            loading.classList.add('hidden');
            container.classList.remove('hidden');
        }
    }

    setupQueryForm() {
        const form = document.getElementById('query-form');
        const input = document.getElementById('query-input');
        const results = document.getElementById('query-results');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const query = input.value.trim();
            if (!query) return;

            results.innerHTML = '<div class="flex items-center justify-center py-4"><div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div><span class="ml-2">Executing query...</span></div>';

            try {
                const result = await window.tftAPI.executeQuery(query);
                results.innerHTML = window.tftAPI.formatQueryResults(result);
            } catch (error) {
                results.innerHTML = `
                    <div class="bg-red-50 border border-red-200 rounded p-3">
                        <h4 class="text-red-800 font-medium">Query Error</h4>
                        <p class="text-red-700 text-sm mt-1">${error.message}</p>
                    </div>
                `;
            }
        });
    }

    setupExampleQueries() {
        document.querySelectorAll('.example-query').forEach(button => {
            button.addEventListener('click', () => {
                const query = button.getAttribute('data-query');
                document.getElementById('query-input').value = query;
            });
        });
    }

    async loadFunctionStatus() {
        const container = document.getElementById('functions-status');

        container.innerHTML = '<div class="text-center py-4"><div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto"></div><p class="mt-2">Testing functions...</p></div>';

        try {
            const results = await window.tftAPI.testFunctions();

            let html = '';
            Object.entries(results).forEach(([func, result]) => {
                const statusColor = result.status === 'success' ? 'green' : 'red';
                html += `
                    <div class="flex justify-between items-center py-2 border-b">
                        <span class="font-medium">api-${func}</span>
                        <div class="flex items-center space-x-2">
                            <span class="px-2 py-1 rounded text-sm bg-${statusColor}-100 text-${statusColor}-800">
                                ${result.status === 'success' ? '✓ Online' : '✗ Error'}
                            </span>
                            ${result.duration ? `<span class="text-xs text-gray-500">${result.duration}</span>` : ''}
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;
        } catch (error) {
            container.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded p-3">
                    <h4 class="text-red-800 font-medium">Status Check Failed</h4>
                    <p class="text-red-700 text-sm mt-1">${error.message}</p>
                </div>
            `;
        }
    }

    async showClusterDetails(clusterId) {
        // This could open a modal or navigate to a details page
        console.log('Show cluster details for:', clusterId);

        try {
            const details = await window.tftAPI.getClusterDetails(clusterId);
            alert(`Cluster ${clusterId}: ${details.name}\nSize: ${details.size}\nAvg Placement: ${details.avg_placement}`);
        } catch (error) {
            alert(`Error loading cluster details: ${error.message}`);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TFTAnalyticsApp();
});