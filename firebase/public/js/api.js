// TFT Analytics API Client
// Connects Firebase frontend to Cloud Functions

class TFTAnalyticsAPI {
    constructor() {
        // Configure API base URL - update with your project ID
        this.PROJECT_ID = 'your-project-id'; // TODO: Update this
        this.REGION = 'us-central1';
        this.BASE_URL = `https://${this.REGION}-${this.PROJECT_ID}.cloudfunctions.net`;

        // Function endpoints
        this.endpoints = {
            stats: `${this.BASE_URL}/api-stats`,
            clusters: `${this.BASE_URL}/api-clusters`,
            query: `${this.BASE_URL}/api-query`,
            clusterDetails: `${this.BASE_URL}/api-cluster-details`
        };
    }

    // Get database statistics
    async getStats() {
        try {
            const response = await fetch(this.endpoints.stats);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch stats:', error);
            throw error;
        }
    }

    // Get available clusters
    async getClusters() {
        try {
            const response = await fetch(this.endpoints.clusters);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch clusters:', error);
            throw error;
        }
    }

    // Get cluster details
    async getClusterDetails(clusterId) {
        try {
            const url = `${this.endpoints.clusterDetails}?id=${clusterId}`;
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch cluster details:', error);
            throw error;
        }
    }

    // Execute TFT query
    async executeQuery(queryText) {
        try {
            const response = await fetch(this.endpoints.query, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: queryText,
                    type: 'auto'
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to execute query:', error);
            throw error;
        }
    }

    // Test function connectivity
    async testFunctions() {
        const results = {};
        const functions = ['stats', 'clusters'];

        for (const func of functions) {
            try {
                const start = Date.now();

                if (func === 'stats') {
                    await this.getStats();
                } else if (func === 'clusters') {
                    await this.getClusters();
                }

                const duration = Date.now() - start;
                results[func] = {
                    status: 'success',
                    duration: `${duration}ms`
                };
            } catch (error) {
                results[func] = {
                    status: 'error',
                    error: error.message
                };
            }
        }

        return results;
    }

    // Format statistics for display
    formatStats(stats) {
        return {
            matches: this.formatNumber(stats.matches || 0),
            participants: this.formatNumber(stats.participants || 0),
            avgPlayersPerMatch: (stats.avg_players_per_match || 0).toFixed(1),
            versionscovered: stats.versions_covered || 0,
            avgLevel: (stats.avg_level || 0).toFixed(1),
            avgPlacement: (stats.avg_placement || 0).toFixed(1),
            lastUpdated: stats.last_updated ? new Date(stats.last_updated).toLocaleDateString() : 'Unknown'
        };
    }

    // Format numbers with commas
    formatNumber(num) {
        return num.toLocaleString();
    }

    // Format query results for display
    formatQueryResults(result) {
        if (!result || !result.data) {
            return '<p class="text-gray-500">No data returned</p>';
        }

        if (result.type === 'stats') {
            const stats = result.data;
            return `
                <div class="space-y-2">
                    <h3 class="font-semibold">Query Statistics</h3>
                    <div class="grid grid-cols-2 gap-4 text-sm">
                        <div><strong>Play Count:</strong> ${this.formatNumber(stats.play_count || 0)}</div>
                        <div><strong>Avg Placement:</strong> ${(stats.avg_placement || 0).toFixed(2)}</div>
                        <div><strong>Win Rate:</strong> ${(stats.winrate || 0).toFixed(1)}%</div>
                        <div><strong>Top 4 Rate:</strong> ${(stats.top4_rate || 0).toFixed(1)}%</div>
                    </div>
                </div>
            `;
        } else if (result.type === 'participants') {
            const participants = result.data.slice(0, 10); // Show first 10
            let html = `<h3 class="font-semibold mb-2">Participants (showing ${participants.length} of ${result.count})</h3>`;
            html += '<div class="overflow-x-auto"><table class="min-w-full text-sm">';
            html += '<thead><tr class="border-b"><th class="text-left p-2">Place</th><th class="text-left p-2">Level</th><th class="text-left p-2">Round</th><th class="text-left p-2">Units</th></tr></thead>';
            html += '<tbody>';

            participants.forEach(p => {
                const units = p.units ? p.units.slice(0, 3).map(u => u.character_id || u.name || 'Unknown').join(', ') : 'None';
                html += `<tr class="border-b"><td class="p-2">${p.placement}</td><td class="p-2">${p.level}</td><td class="p-2">${p.last_round}</td><td class="p-2">${units}</td></tr>`;
            });

            html += '</tbody></table></div>';
            return html;
        }

        return `<pre class="text-sm overflow-x-auto">${JSON.stringify(result.data, null, 2)}</pre>`;
    }
}

// Global API instance
window.tftAPI = new TFTAnalyticsAPI();