/**
 * Morpheus Research Lab - UI Controller
 * ======================================
 * Read-only visibility panel for SuperBot Research Framework.
 *
 * SAFETY: This UI does NOT connect to:
 * - Live order endpoints
 * - Execution endpoints
 * - Production config modification
 *
 * Version: 1.1
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const API = {
    RESEARCH: '/api/research',
    SUPERBOT: '/api/superbot'
};

// Refresh interval (30 seconds)
const REFRESH_INTERVAL = 30000;

// State
let refreshTimer = null;
let selectedRunId = null;
let equityChart = null;

// ═══════════════════════════════════════════════════════════════════════════════
// API HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchAPI(endpoint, base = API.RESEARCH) {
    try {
        const response = await fetch(`${base}${endpoint}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        return null;
    }
}

async function postAPI(endpoint, data = {}, base = API.SUPERBOT) {
    try {
        const response = await fetch(`${base}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        return null;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION A: Research Status Dashboard
// ═══════════════════════════════════════════════════════════════════════════════

async function updateStatusDashboard() {
    const data = await fetchAPI('/status');

    if (!data) {
        setConnectionStatus(false);
        return;
    }

    setConnectionStatus(true);

    // Update stats
    document.getElementById('lastIngestion').textContent = data.last_ingestion || 'Never';
    document.getElementById('signalsIngested').textContent = data.signals_ingested || 0;
    document.getElementById('tradesIngested').textContent = data.trades_ingested || 0;
    document.getElementById('regimesIngested').textContent = data.regimes_ingested || 0;
    document.getElementById('dataHash').textContent = data.data_hash || '--';
    document.getElementById('gitSha').textContent = data.git_sha || '--';
    document.getElementById('lastShadowRun').textContent = data.last_shadow_run_id || 'None';

    // Replay accuracy with color coding
    const accuracy = data.replay_accuracy_pct || 0;
    const accuracyEl = document.getElementById('replayAccuracy');
    accuracyEl.textContent = `${accuracy.toFixed(1)}%`;
    accuracyEl.className = accuracy >= 99 ? 'stat-value success' :
                           accuracy >= 95 ? 'stat-value warning' : 'stat-value danger';

    // Warning banner
    const warningBanner = document.getElementById('integrityWarning');
    if (data.integrity_warning) {
        warningBanner.style.display = 'flex';
        document.getElementById('warningMessage').textContent = data.warning_message;
    } else {
        warningBanner.style.display = 'none';
    }

    // Update timestamp
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
}

function setConnectionStatus(connected) {
    const dot = document.getElementById('connectionDot');
    const text = document.getElementById('connectionText');

    if (connected) {
        dot.className = 'status-dot connected';
        text.textContent = 'Connected';
    } else {
        dot.className = 'status-dot disconnected';
        text.textContent = 'Disconnected';
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION B: Replay Integrity Panel
// ═══════════════════════════════════════════════════════════════════════════════

async function updateIntegrityPanel() {
    const data = await fetchAPI('/replay/integrity');

    if (!data) return;

    document.getElementById('prodTradeCount').textContent = data.production_trade_count || 0;
    document.getElementById('replayTradeCount').textContent = data.replay_trade_count || 0;

    // Match percentage
    const matchPct = data.matched_trades_pct || 0;
    const matchEl = document.getElementById('matchedPct');
    matchEl.textContent = `${matchPct.toFixed(1)}%`;
    matchEl.className = getStatusClass(matchPct, 99, 90);

    // Entry variance
    const entryVar = data.entry_price_variance || 0;
    const entryEl = document.getElementById('entryVariance');
    entryEl.textContent = `${(entryVar * 100).toFixed(2)}%`;
    entryEl.className = entryVar < 0.05 ? 'stat-value success' :
                        entryVar < 0.1 ? 'stat-value warning' : 'stat-value danger';

    // Exit variance
    const exitVar = data.exit_price_variance || 0;
    const exitEl = document.getElementById('exitVariance');
    exitEl.textContent = `${(exitVar * 100).toFixed(2)}%`;
    exitEl.className = exitVar < 0.05 ? 'stat-value success' :
                       exitVar < 0.1 ? 'stat-value warning' : 'stat-value danger';

    // Status indicator
    const statusEl = document.getElementById('integrityStatus');
    statusEl.textContent = data.status || 'UNKNOWN';
    statusEl.className = `status-badge ${data.status_color || 'neutral'}`;
}

function getStatusClass(value, goodThreshold, warnThreshold) {
    if (value >= goodThreshold) return 'stat-value success';
    if (value >= warnThreshold) return 'stat-value warning';
    return 'stat-value danger';
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION C: Shadow Runs Table
// ═══════════════════════════════════════════════════════════════════════════════

async function updateRunsTable() {
    const data = await fetchAPI('/runs');

    if (!data || !data.runs) {
        document.getElementById('runsTableBody').innerHTML = `
            <tr><td colspan="8" class="empty-cell">No shadow runs available</td></tr>
        `;
        document.getElementById('runsCount').textContent = '0 runs';
        return;
    }

    document.getElementById('runsCount').textContent = `${data.runs.length} runs`;

    const tbody = document.getElementById('runsTableBody');
    tbody.innerHTML = data.runs.slice(0, 50).map(run => `
        <tr class="clickable-row" onclick="openRunModal('${run.run_id}')">
            <td class="mono">${run.run_id || '--'}</td>
            <td class="mono">${run.config_hash || '--'}</td>
            <td>${run.date_range || 'N/A'}</td>
            <td class="${run.expectancy >= 0 ? 'positive' : 'negative'}">${run.expectancy.toFixed(4)}</td>
            <td class="negative">$${run.max_drawdown.toFixed(2)}</td>
            <td>${run.trade_count}</td>
            <td>${renderWFBadge(run.wf_pass)}</td>
            <td>${formatTimestamp(run.created)}</td>
        </tr>
    `).join('');
}

function renderWFBadge(status) {
    if (status === null || status === undefined) {
        return '<span class="badge neutral">--</span>';
    }
    return status ?
        '<span class="badge success">PASS</span>' :
        '<span class="badge danger">FAIL</span>';
}

function formatTimestamp(ts) {
    if (!ts) return '--';
    try {
        return new Date(ts).toLocaleString();
    } catch {
        return ts;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION C: Run Detail Modal
// ═══════════════════════════════════════════════════════════════════════════════

async function openRunModal(runId) {
    selectedRunId = runId;

    const modal = document.getElementById('runModal');
    const content = document.getElementById('modalContent');

    // Show loading state
    modal.style.display = 'flex';
    content.innerHTML = '<div class="loading-spinner"></div>';

    // Fetch run details
    const runData = await fetchAPI(`/runs/${runId}`);

    if (!runData) {
        content.innerHTML = '<div class="error-message">Failed to load run data</div>';
        return;
    }

    // Build modal content
    content.innerHTML = `
        <div class="modal-header">
            <h2>Shadow Run: ${runId}</h2>
            <button class="close-btn" onclick="closeModal()">&times;</button>
        </div>

        <div class="modal-body">
            <!-- Metrics Grid -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${runData.metrics?.trade_count || 0}</div>
                    <div class="metric-label">Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value ${runData.metrics?.win_rate >= 0.5 ? 'positive' : 'warning'}">
                        ${((runData.metrics?.win_rate || 0) * 100).toFixed(1)}%
                    </div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value ${runData.metrics?.total_pnl >= 0 ? 'positive' : 'negative'}">
                        $${(runData.metrics?.total_pnl || 0).toFixed(2)}
                    </div>
                    <div class="metric-label">Total PnL</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(runData.metrics?.profit_factor || 0).toFixed(2)}</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
            </div>

            <!-- Equity Curve Chart -->
            <div class="chart-container">
                <h3>Equity Curve</h3>
                <canvas id="equityCurveChart"></canvas>
            </div>

            <!-- Regime Breakdown -->
            <div class="regime-breakdown">
                <h3>Regime Performance</h3>
                <div class="regime-grid" id="regimeGrid">
                    ${renderRegimeBreakdown(runData.regime_breakdown)}
                </div>
            </div>

            <!-- Diff vs Production -->
            <div class="diff-section">
                <h3>Diff vs Production</h3>
                <div class="diff-grid">
                    <div class="diff-item">
                        <span class="diff-label">Expectancy Delta:</span>
                        <span class="diff-value ${runData.diff_vs_production?.expectancy_delta >= 0 ? 'positive' : 'negative'}">
                            ${runData.diff_vs_production?.expectancy_delta >= 0 ? '+' : ''}${runData.diff_vs_production?.expectancy_delta?.toFixed(4) || '--'}
                        </span>
                    </div>
                    <div class="diff-item">
                        <span class="diff-label">Win Rate Delta:</span>
                        <span class="diff-value ${runData.diff_vs_production?.win_rate_delta >= 0 ? 'positive' : 'negative'}">
                            ${runData.diff_vs_production?.win_rate_delta >= 0 ? '+' : ''}${runData.diff_vs_production?.win_rate_delta?.toFixed(1) || '--'}%
                        </span>
                    </div>
                    <div class="diff-item">
                        <span class="diff-label">Profit Factor Delta:</span>
                        <span class="diff-value ${runData.diff_vs_production?.profit_factor_delta >= 0 ? 'positive' : 'negative'}">
                            ${runData.diff_vs_production?.profit_factor_delta >= 0 ? '+' : ''}${runData.diff_vs_production?.profit_factor_delta?.toFixed(2) || '--'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Render equity curve chart
    renderEquityCurve(runData.equity_curve || [], runData.equity_curve_labels || []);
}

function renderRegimeBreakdown(regimes) {
    if (!regimes || Object.keys(regimes).length === 0) {
        return '<div class="no-data">No regime data available</div>';
    }

    return Object.entries(regimes).map(([regime, data]) => `
        <div class="regime-card">
            <div class="regime-name">${regime}</div>
            <div class="regime-stats">
                <span>Trades: ${data.trade_count}</span>
                <span class="${data.win_rate >= 0.5 ? 'positive' : 'warning'}">WR: ${(data.win_rate * 100).toFixed(1)}%</span>
                <span class="${data.total_pnl >= 0 ? 'positive' : 'negative'}">$${data.total_pnl.toFixed(2)}</span>
            </div>
        </div>
    `).join('');
}

function renderEquityCurve(data, labels) {
    const ctx = document.getElementById('equityCurveChart');
    if (!ctx) return;

    // Destroy existing chart
    if (equityChart) {
        equityChart.destroy();
    }

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Equity ($)',
                data: data,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#666' }
                },
                y: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: {
                        color: '#666',
                        callback: (v) => `$${v.toFixed(0)}`
                    }
                }
            }
        }
    });
}

function closeModal() {
    document.getElementById('runModal').style.display = 'none';
    selectedRunId = null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION D: Walk-Forward Results
// ═══════════════════════════════════════════════════════════════════════════════

async function updateWalkForwardPanel() {
    // Get latest validation
    const validations = await fetchAPI('/validations', API.SUPERBOT);

    if (!validations || !validations.validations || validations.validations.length === 0) {
        document.getElementById('wfContent').innerHTML = `
            <div class="no-data">No walk-forward validations run yet</div>
        `;
        return;
    }

    const latest = validations.validations[validations.validations.length - 1];
    const summary = latest.summary || {};

    // Calculate metrics
    const isPnl = summary.total_is_pnl || 0;
    const oosPnl = summary.total_oos_pnl || 0;
    const degradation = isPnl > 0 ? ((isPnl - oosPnl) / isPnl * 100) : 0;
    const stability = 1 - Math.min(1, Math.abs(degradation) / 100);

    const degradationWarning = degradation > 30;

    document.getElementById('wfContent').innerHTML = `
        <div class="wf-grid">
            <div class="wf-stat">
                <div class="wf-value positive">$${isPnl.toFixed(2)}</div>
                <div class="wf-label">In-Sample Expectancy</div>
            </div>
            <div class="wf-stat">
                <div class="wf-value ${oosPnl >= 0 ? 'positive' : 'negative'}">$${oosPnl.toFixed(2)}</div>
                <div class="wf-label">Out-of-Sample Expectancy</div>
            </div>
            <div class="wf-stat">
                <div class="wf-value ${degradationWarning ? 'danger' : 'success'}">${degradation.toFixed(1)}%</div>
                <div class="wf-label">Degradation</div>
            </div>
            <div class="wf-stat">
                <div class="wf-value">${stability.toFixed(2)}</div>
                <div class="wf-label">Stability Score</div>
            </div>
        </div>

        <div class="wf-indicator ${latest.is_valid ? 'pass' : 'fail'}">
            ${latest.is_valid ? 'PASS' : 'FAIL'}
        </div>

        ${degradationWarning ? `
            <div class="wf-warning">
                <span class="warning-icon">⚠️</span>
                Degradation > 30% indicates possible overfitting. Review before promoting.
            </div>
        ` : ''}

        ${latest.overfit_detected ? `
            <div class="wf-warning danger">
                <span class="warning-icon">🚨</span>
                OVERFIT DETECTED - Do not promote this configuration.
            </div>
        ` : ''}
    `;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION E: Proposal Review Queue
// ═══════════════════════════════════════════════════════════════════════════════

async function updateProposalQueue() {
    const data = await fetchAPI('/proposals', API.SUPERBOT);

    if (!data || !data.proposals || data.proposals.length === 0) {
        document.getElementById('proposalTableBody').innerHTML = `
            <tr><td colspan="6" class="empty-cell">No proposals in queue</td></tr>
        `;
        document.getElementById('proposalCount').textContent = '0 pending';
        return;
    }

    const pending = data.proposals.filter(p => p.status === 'PENDING_REVIEW');
    document.getElementById('proposalCount').textContent = `${pending.length} pending`;

    const tbody = document.getElementById('proposalTableBody');
    tbody.innerHTML = data.proposals.map(p => `
        <tr>
            <td class="mono">${p.proposal_id}</td>
            <td class="positive">+${p.improvement_pct || '--'}%</td>
            <td>${p.drawdown_delta || '--'}</td>
            <td>${renderWFBadge(p.walk_forward_valid)}</td>
            <td>${p.sample_size || '--'}</td>
            <td>
                <span class="badge ${getStatusBadgeClass(p.status)}">${p.status}</span>
            </td>
            <td class="action-cell">
                <button class="btn btn-sm btn-info" onclick="viewEvidence('${p.proposal_id}')">Evidence</button>
                ${p.status === 'PENDING_REVIEW' ? `
                    <button class="btn btn-sm btn-success" onclick="approveProposal('${p.proposal_id}')"
                            ${!p.walk_forward_valid ? 'disabled title="WF invalid"' : ''}>Approve</button>
                    <button class="btn btn-sm btn-danger" onclick="rejectProposal('${p.proposal_id}')">Reject</button>
                ` : ''}
            </td>
        </tr>
    `).join('');
}

function getStatusBadgeClass(status) {
    switch (status) {
        case 'APPROVED': return 'success';
        case 'REJECTED': return 'danger';
        case 'PENDING_REVIEW': return 'warning';
        default: return 'neutral';
    }
}

async function viewEvidence(proposalId) {
    const data = await fetchAPI(`/proposals/${proposalId}/evidence`);

    if (!data) {
        alert('Failed to load evidence');
        return;
    }

    const modal = document.getElementById('runModal');
    const content = document.getElementById('modalContent');

    modal.style.display = 'flex';
    content.innerHTML = `
        <div class="modal-header">
            <h2>Evidence: ${proposalId}</h2>
            <button class="close-btn" onclick="closeModal()">&times;</button>
        </div>

        <div class="modal-body">
            <div class="evidence-section">
                <h3>Risk Assessment</h3>
                <div class="evidence-item">
                    <span>Risk Level:</span>
                    <span class="badge ${data.risk_level === 'LOW' ? 'success' : data.risk_level === 'MEDIUM' ? 'warning' : 'danger'}">
                        ${data.risk_level || 'UNKNOWN'}
                    </span>
                </div>
                <div class="evidence-item">
                    <span>Walk-Forward Valid:</span>
                    <span class="badge ${data.walk_forward_valid ? 'success' : 'danger'}">
                        ${data.walk_forward_valid ? 'YES' : 'NO'}
                    </span>
                </div>
            </div>

            <div class="evidence-section">
                <h3>Parameter Changes</h3>
                <pre class="code-block">${JSON.stringify(data.diff_vs_production?.parameter_changes || {}, null, 2)}</pre>
            </div>

            <div class="evidence-section">
                <h3>Performance Comparison</h3>
                <pre class="code-block">${JSON.stringify(data.evidence || {}, null, 2)}</pre>
            </div>
        </div>
    `;
}

async function approveProposal(proposalId) {
    // Confirmation dialog
    const confirmed = confirm(
        `APPROVE PROPOSAL: ${proposalId}\n\n` +
        `This marks the proposal for production adoption by Supervisor.\n\n` +
        `No automatic promotion will occur - manual Supervisor approval required.\n\n` +
        `Continue?`
    );

    if (!confirmed) return;

    const result = await postAPI(`/proposals/${proposalId}/approve`);

    if (result && result.success) {
        alert(`Proposal ${proposalId} approved.\n\nSupervisor review required for production adoption.`);
        updateProposalQueue();
    } else {
        alert('Failed to approve proposal.');
    }
}

async function rejectProposal(proposalId) {
    const confirmed = confirm(`Reject proposal ${proposalId}?`);

    if (!confirmed) return;

    const result = await postAPI(`/proposals/${proposalId}/reject`);

    if (result && result.success) {
        alert(`Proposal ${proposalId} rejected.`);
        updateProposalQueue();
    } else {
        alert('Failed to reject proposal.');
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REFRESH & INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

async function refreshAll() {
    console.log('Refreshing all data...');

    await Promise.all([
        updateStatusDashboard(),
        updateIntegrityPanel(),
        updateRunsTable(),
        updateWalkForwardPanel(),
        updateProposalQueue()
    ]);
}

function startAutoRefresh() {
    refreshTimer = setInterval(refreshAll, REFRESH_INTERVAL);
}

function stopAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Research Lab UI initializing...');
    refreshAll();
    startAutoRefresh();
});

// Cleanup
window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
});

// Close modal on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// Close modal on backdrop click
document.addEventListener('click', (e) => {
    const modal = document.getElementById('runModal');
    if (e.target === modal) {
        closeModal();
    }
});
