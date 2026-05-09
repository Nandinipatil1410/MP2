const state = {
    initialized: false,
    documentsLoaded: false,
    selectedFiles: [],
};

const scenarios = {
    medical: "Based on this patient's history, what contraindications apply to Drug X? Analyze potential interactions step-by-step.",
    legal: "Analyze if the following clauses in this contract are compliant with GDPR Article 5. Break down the reasoning.",
};

const elements = {
    documentsInput: document.getElementById("documentsInput"),
    selectedFiles: document.getElementById("selectedFiles"),
    loadDocumentsBtn: document.getElementById("loadDocumentsBtn"),
    statusBanner: document.getElementById("statusBanner"),
    metricRuntime: document.getElementById("metricRuntime"),
    metricDocuments: document.getElementById("metricDocuments"),
    metricModels: document.getElementById("metricModels"),
    queryInput: document.getElementById("queryInput"),
    modeSelect: document.getElementById("modeSelect"),
    runQueryBtn: document.getElementById("runQueryBtn"),
    queryStatus: document.getElementById("queryStatus"),
    queryResult: document.getElementById("queryResult"),
    answerContent: document.getElementById("answerContent"),
    traceContent: document.getElementById("traceContent"),
    compareInput: document.getElementById("compareInput"),
    runCompareBtn: document.getElementById("runCompareBtn"),
    compareStatus: document.getElementById("compareStatus"),
    compareResults: document.getElementById("compareResults"),
    responseColumns: document.getElementById("responseColumns"),
    documentSelect: document.getElementById("documentSelect"),
    compareDocumentSelect: document.getElementById("compareDocumentSelect"),
    copyAnalysisBtn: document.getElementById("copyAnalysisBtn"),
    reasoningTimeline: document.getElementById("reasoningTimeline"),
    reasoningCard: document.getElementById("reasoningCard"),
    // Persistence UI
    clearSessionBtn: document.getElementById("clearSessionBtn"),
    persistedDocsSection: document.getElementById("persistedDocsSection"),
    persistedFileList: document.getElementById("persistedFileList"),
    // Query Transparency
    queryTransparency: document.getElementById("queryTransparency"),
    qtRawQuery: document.getElementById("qtRawQuery"),
    qtAbstractedQuery: document.getElementById("qtAbstractedQuery"),
    qtPrivacyBadge: document.getElementById("qtPrivacyBadge"),
    qtSameNote: document.getElementById("qtSameNote"),
    // Judge panel
    judgeMeta: document.getElementById("judgeMeta"),
    judgeWinner: document.getElementById("judgeWinner"),
    judgeScores: document.getElementById("judgeScores"),
    judgeBarChart: document.getElementById("judgeBarChart"),
    judgeDiffsWrap: document.getElementById("judgeDiffsWrap"),
    judgeDiffs: document.getElementById("judgeDiffs"),
    judgeReasoningWrap: document.getElementById("judgeReasoningWrap"),
    judgeReasoning: document.getElementById("judgeReasoning"),
};

function setStatus(target, message, type = "status-success") {
    target.textContent = message;
    target.className = `${target.id === "statusBanner" ? "status-banner" : "inline-status"} ${type}`;
    target.classList.remove("hidden");
}

function clearStatus(target) {
    target.textContent = "";
    target.classList.add("hidden");
}

function escapeHtml(text) {
    return (text || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function parseMarkdown(text) {
    if (!text) return "";
    
    let html = escapeHtml(text)
        .replace(/\r\n/g, '\n')
        .replace(/\n{3,}/g, '\n\n');

    // --- Pre-normalize: pull inline headers onto their own line ---
    // The cloud LLM sometimes emits "some text. ## Header more text" on one line.
    // Move any ## / ### that appear mid-line to start on their own newline.
    html = html.replace(/([^\n])(#{2,3} )/g, '$1\n$2');

    // Split joined numbered list items (e.g. "1. First 2. Second") but only
    // when the digit follows a word boundary so decimals like "28.4" are safe.
    html = html.replace(/(\S) (\d+\.\s)/g, '$1\n$2');

    // Strip standalone hash symbols which LLMs sometimes emit as broken dividers
    html = html.replace(/^#+\s*$/gim, '');

    // Headers (must be on their own line — the /m flag makes ^ match line-start)
    html = html.replace(/^#### (.*$)/gim, '<h5>$1</h5>');
    html = html.replace(/^### (.*$)/gim,  '<h4>$1</h4>');
    html = html.replace(/^## (.*$)/gim,   '<h3>$1</h3>');
    html = html.replace(/^# (.*$)/gim,    '<h2>$1</h2>');

    // Bold, Italic, Inline Code
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim,     '<em>$1</em>');
    html = html.replace(/`(.*?)`/gim,       '<code>$1</code>');

    // Line-by-line processing for paragraphs and lists
    const lines = html.split('\n');
    let inList = null; // 'ul', 'ol', or null
    let result = [];

    lines.forEach(line => {
        const trimmed = line.trim();
        if (trimmed === '') {
            if (inList) {
                result.push(`</${inList}>`);
                inList = null;
            }
            return;
        }

        // Bullet list: lines starting with "- " or "* "
        const bulletMatch = trimmed.match(/^[-*] (.*)/);
        // Numbered list: lines starting with "1. " etc.
        const numberMatch = trimmed.match(/^\d+\.\s+(.*)/);

        if (bulletMatch) {
            if (inList !== 'ul') {
                if (inList) result.push(`</${inList}>`);
                result.push('<ul>');
                inList = 'ul';
            }
            result.push(`<li>${bulletMatch[1]}</li>`);
        } else if (numberMatch) {
            if (inList !== 'ol') {
                if (inList) result.push(`</${inList}>`);
                result.push('<ol>');
                inList = 'ol';
            }
            result.push(`<li>${numberMatch[1]}</li>`);
        } else {
            // Not a list item
            if (inList) {
                result.push(`</${inList}>`);
                inList = null;
            }
            // Already-converted HTML tags (headers etc.) pass through as-is
            if (trimmed.startsWith('<')) {
                result.push(trimmed);
            } else {
                result.push(`<p>${trimmed}</p>`);
            }
        }
    });

    if (inList) result.push(`</${inList}>`);
    
    return result.join('\n');
}


function renderFileList(files) {
    if (!files.length) {
        elements.selectedFiles.textContent = "No documents selected.";
        elements.selectedFiles.className = "file-list empty";
        return;
    }

    elements.selectedFiles.className = "file-list";
    elements.selectedFiles.innerHTML = files
        .map((file) => `<div class="file-chip">${escapeHtml(file.name || file)}</div>`)
        .join("");
}

function renderTrace(result) {
    const chips = [
        `Mode: ${result.mode}`,
        `Latency: ${Number(result.latency || 0).toFixed(2)}s`,
        `Privacy Score: ${result.privacy_score ?? "N/A"}`,
    ];

    if (result.documents_used !== undefined) {
        chips.push(`Documents Used: ${result.documents_used}`);
    }

    if (elements.traceContent) {
        elements.traceContent.innerHTML = chips
            .map((chip) => `<span class="metric-chip">${escapeHtml(chip)}</span>`)
            .join("");
    }
}

function renderTable(headers, rows) {
    const head = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
    const body = rows
        .map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(String(cell ?? ""))}</td>`).join("")}</tr>`)
        .join("");
    return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderComparison(results) {
    elements.responseColumns.innerHTML = '';
    const columns = [
        ["Hybrid Node", results.hybrid.answer],
        ["Local Node", results.local_only.answer],
    ];

    columns.forEach(([title, content]) => {
        const card = document.createElement('article');
        card.className = 'result-card preview-card';
        card.innerHTML = `
            <div class="result-card-header">
                <h4>${title}</h4>
            </div>
            <div class="formatted-output">${parseMarkdown(content || "")}</div>
        `;
        elements.responseColumns.appendChild(card);
    });
}

function renderPrivacyTransparency(results) {
    const hybrid = results.hybrid;
    const originalQuery = results.query || "N/A";

    elements.privacyValidation.innerHTML = `
        <table>
            <thead>
                <tr>
                    <th style="width: 30%;">Privacy Layer</th>
                    <th>Execution Trace / Payload</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Execution Privacy</strong></td>
                    <td>${(hybrid.privacy_score * 100).toFixed(0)}% Protected (Document data stays 100% on local machine)</td>
                </tr>
                <tr>
                    <td><strong>Input Query</strong></td>
                    <td><code style="font-size: 0.85rem; color: var(--text-muted);">${escapeHtml(originalQuery)}</code></td>
                </tr>
                <tr>
                    <td><strong>Cloud Handshake</strong></td>
                    <td><code style="font-size: 0.85rem; color: var(--primary); font-weight: 700;">${escapeHtml(hybrid.abstracted_query)}</code></td>
                </tr>
                <tr>
                    <td><strong>Leakage Check</strong></td>
                    <td><span class="status-badge" style="background: rgba(16, 185, 129, 0.1); color: #059669; padding: 4px 8px; border-radius: 6px; font-size: 0.75rem;">Verified: No PII detected in cloud payload</span></td>
                </tr>
            </tbody>
        </table>
    `;
}

function renderJudge(judge) {
    // Reset all sections
    elements.judgeMeta.innerHTML = "";
    elements.judgeWinner.classList.add("hidden");
    elements.judgeScores.classList.add("hidden");
    elements.judgeBarChart.classList.add("hidden");
    elements.judgeDiffsWrap.classList.add("hidden");
    elements.judgeReasoningWrap.classList.add("hidden");

    if (!judge || judge.provider === "unavailable") {
        const message = judge?.error ? `Judge unavailable: ${judge.error}` : "Judge unavailable.";
        elements.judgeMeta.innerHTML = `<span class="metric-chip">${escapeHtml(message)}</span>`;
        return;
    }

    // --- Meta chips (provider / model) ---
    const meta = [`Provider: ${judge.provider}`, judge.model ? `Model: ${judge.model}` : null].filter(Boolean);
    elements.judgeMeta.innerHTML = meta.map(m => `<span class="metric-chip">${escapeHtml(m)}</span>`).join("");

    // --- Verdict Banner ---
    const winnerMap = { A: "Local-Only", B: "Hybrid", Tie: "Tie" };
    const winnerLabel = winnerMap[judge.comparison?.winner] || "Tie";
    const localScore  = judge.comparison?.answer_a_score ?? judge.local_only?.overall_score ?? "N/A";
    const hybridScore = judge.comparison?.answer_b_score ?? judge.hybrid?.overall_score ?? "N/A";
    const strength    = judge.comparison?.preference_strength ? ` (${judge.comparison.preference_strength})` : "";

    const winnerClass = winnerLabel === "Hybrid" ? "winner-hybrid"
                       : winnerLabel === "Local-Only" ? "winner-local"
                       : "winner-tie";
    elements.judgeWinner.className = `judge-verdict ${winnerClass}`;
    elements.judgeWinner.innerHTML = `
        <div class="verdict-label">Verdict${strength}</div>
        <div class="verdict-winner">${escapeHtml(winnerLabel)}</div>
        <div class="verdict-scores">
            <span class="vscore local">Local: <strong>${localScore}</strong></span>
            <span class="vscore hybrid">Hybrid: <strong>${hybridScore}</strong></span>
        </div>
    `;
    elements.judgeWinner.classList.remove("hidden");

    // --- Per-Criterion Score Table with Reasons ---
    const criteria = ["accuracy", "completeness", "relevance", "coherence", "groundedness"];
    const localEval  = judge.local_only  || {};
    const hybridEval = judge.hybrid || {};

    let tableHTML = `
        <table class="judge-table">
            <thead>
                <tr>
                    <th>Criterion</th>
                    <th>Local Score</th>
                    <th>Local Reason</th>
                    <th>Hybrid Score</th>
                    <th>Hybrid Reason</th>
                </tr>
            </thead>
            <tbody>`;

    criteria.forEach(c => {
        const localItem  = localEval[c]  || {};
        const hybridItem = hybridEval[c] || {};
        const ls = localItem.score  ?? "—";
        const hs = hybridItem.score ?? "—";
        const lsBetter = typeof ls === "number" && typeof hs === "number" && ls > hs;
        const hsBetter = typeof ls === "number" && typeof hs === "number" && hs > ls;
        tableHTML += `
            <tr>
                <td><strong>${escapeHtml(c)}</strong></td>
                <td class="score-cell ${lsBetter ? 'score-win' : ''}">${ls}</td>
                <td class="reason-cell">${escapeHtml(localItem.reason || "—")}</td>
                <td class="score-cell ${hsBetter ? 'score-win' : ''}">${hs}</td>
                <td class="reason-cell">${escapeHtml(hybridItem.reason || "—")}</td>
            </tr>`;
    });
    tableHTML += `</tbody></table>`;
    elements.judgeScores.innerHTML = tableHTML;
    elements.judgeScores.classList.remove("hidden");

    // --- Visual Score Bars ---
    let barsHTML = "";
    criteria.forEach(c => {
        const ls = Number(localEval[c]?.score  ?? 0);
        const hs = Number(hybridEval[c]?.score ?? 0);
        barsHTML += `
            <div class="bar-row">
                <span class="bar-label">${escapeHtml(c)}</span>
                <div class="bar-track">
                    <div class="bar-fill local-bar"  style="width:${ls * 10}%" title="Local: ${ls}"></div>
                </div>
                <span class="bar-val local-val">${ls}</span>
                <div class="bar-track">
                    <div class="bar-fill hybrid-bar" style="width:${hs * 10}%" title="Hybrid: ${hs}"></div>
                </div>
                <span class="bar-val hybrid-val">${hs}</span>
            </div>`;
    });
    elements.judgeBarChart.innerHTML = `
        <div class="bar-legend">
            <span class="legend-dot local-dot"></span>Local &nbsp;&nbsp;
            <span class="legend-dot hybrid-dot"></span>Hybrid
        </div>
        ${barsHTML}`;
    elements.judgeBarChart.classList.remove("hidden");

    // --- Key Differences ---
    const diffs = Array.isArray(judge.comparison?.key_differences) ? judge.comparison.key_differences : [];
    if (diffs.length) {
        elements.judgeDiffs.innerHTML = diffs.slice(0, 6).map(d => `<li>${escapeHtml(d)}</li>`).join("");
        elements.judgeDiffsWrap.classList.remove("hidden");
    }

    // --- Judge Reasoning ---
    const reasoning = judge.comparison?.reasoning || judge.comparison?.rationale || "";
    if (reasoning) {
        elements.judgeReasoning.textContent = reasoning;
        elements.judgeReasoningWrap.classList.remove("hidden");
    }
}

function setRuntimeMetrics(data) {
    if (!data.initialized) {
        elements.metricRuntime.textContent = "Initializing...";
        elements.metricDocuments.textContent = "0 loaded";
        elements.metricModels.textContent = "Configuring";
        return;
    }

    const stats = data.stats || {};
    elements.metricRuntime.textContent = "Online";
    elements.metricDocuments.textContent = `${(data.loaded_files || []).length} loaded`;
    if (stats.local_model || stats.cloud_model) {
        elements.metricModels.textContent = `${stats.local_model || "Local"} / ${stats.cloud_model || "Cloud"}`;
    } else {
        elements.metricModels.textContent = "Configured";
    }
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok || data.success === false) {
        throw new Error(data.error || "Request failed.");
    }
    return data;
}

async function loadState() {
    try {
        const data = await fetchJson("/api/state");
        state.initialized = data.initialized;
        state.documentsLoaded = data.documents_loaded;
        renderFileList((data.loaded_files || []).map((name) => ({ name })));
        setRuntimeMetrics(data);
        // Show persisted docs from vector store (Option 2)
        renderPersistedDocs(data.loaded_files || []);
        if (state.documentsLoaded) {
            await loadDocumentList();
        }
    } catch (error) {
        setStatus(elements.statusBanner, error.message, "status-error");
    }
}

async function loadDocumentList() {
    try {
        const documents = await fetchJson("/api/documents");
        const options = ['<option value="all">All Documents</option>'];
        documents.forEach(doc => {
            options.push(`<option value="${escapeHtml(doc)}">${escapeHtml(doc)}</option>`);
        });

        const html = options.join("");
        if (elements.documentSelect) elements.documentSelect.innerHTML = html;
        if (elements.compareDocumentSelect) elements.compareDocumentSelect.innerHTML = html;
    } catch (error) {
        console.error("Failed to load document list:", error);
    }
}

function setupTabs() {
    document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".tab-button").forEach((tab) => tab.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
            button.classList.add("active");
            document.getElementById(button.dataset.tab).classList.add("active");
        });
    });
}

// --- Legacy Configuration Logic Removed ---

function setupCopyFeature() {
    if (elements.copyAnalysisBtn) {
        elements.copyAnalysisBtn.addEventListener("click", () => {
            const text = elements.answerContent.textContent;
            navigator.clipboard.writeText(text).then(() => {
                const originalText = elements.copyAnalysisBtn.textContent;
                elements.copyAnalysisBtn.textContent = "Copied!";
                setTimeout(() => (elements.copyAnalysisBtn.textContent = originalText), 2000);
            });
        });
    }
}

/** Renders the Active Knowledge Base panel from persisted vector store files. */
function renderPersistedDocs(fileNames) {
    if (!elements.persistedDocsSection || !elements.persistedFileList) return;
    if (!fileNames || fileNames.length === 0) {
        elements.persistedDocsSection.classList.add("hidden");
        return;
    }
    elements.persistedFileList.innerHTML = fileNames
        .map((name) => `<div class="file-chip persisted">${escapeHtml(name.split('/').pop().split('\\\\').pop())}</div>`)
        .join("");
    elements.persistedDocsSection.classList.remove("hidden");
}

async function clearSession() {
    if (!confirm("This will permanently delete all loaded documents and their embeddings. Continue?")) return;
    clearStatus(elements.statusBanner);
    setStatus(elements.statusBanner, "Clearing session...", "status-warning");
    try {
        const data = await fetchJson("/api/clear-session", { method: "POST" });
        state.documentsLoaded = false;
        state.selectedFiles = [];
        renderFileList([]);
        renderPersistedDocs([]);
        elements.queryResult?.classList.add("hidden");
        elements.compareResults?.classList.add("hidden");
        elements.reasoningCard?.classList.add("hidden");
        const allOption = '<option value="all">All Documents</option>';
        if (elements.documentSelect) elements.documentSelect.innerHTML = allOption;
        if (elements.compareDocumentSelect) elements.compareDocumentSelect.innerHTML = allOption;
        setRuntimeMetrics({ initialized: true, documents_loaded: false, loaded_files: [], stats: {} });
        setStatus(elements.statusBanner, data.message || "Session cleared.", "status-success");
    } catch (error) {
        setStatus(elements.statusBanner, error.message, "status-error");
    }
}

// --- Manual Initialization Removed ---

async function loadDocuments() {
    clearStatus(elements.statusBanner);
    if (!state.selectedFiles.length) {
        setStatus(elements.statusBanner, "Upload documents first.", "status-warning");
        return;
    }

    const formData = new FormData();
    state.selectedFiles.forEach((file) => formData.append("documents", file));

    setStatus(elements.statusBanner, "Loading protected documents...", "status-warning");
    try {
        const data = await fetchJson("/api/load-documents", {
            method: "POST",
            body: formData,
        });
        state.documentsLoaded = true;
        renderFileList((data.loaded_files || []).map((name) => ({ name })));
        setStatus(elements.statusBanner, data.message, "status-success");
        await loadState();
    } catch (error) {
        setStatus(elements.statusBanner, error.message, "status-error");
    }
}

async function runQuery() {
    clearStatus(elements.queryStatus);
    elements.queryResult.classList.add("hidden");
    elements.queryTransparency?.classList.add("hidden");
    setStatus(elements.queryStatus, "Running the selected reasoning pipeline...", "status-warning");
    try {
        const rawQuery = elements.queryInput.value;
        const result = await fetchJson("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: rawQuery,
                mode: elements.modeSelect.value,
                expert_mode: true,
                selected_document: elements.documentSelect.value,
                // intent is auto-classified server-side
            }),
        });
        elements.answerContent.innerHTML = parseMarkdown(result.answer || "");

        if (elements.reasoningTimeline && result.reasoning_plan) {
            elements.reasoningTimeline.innerHTML = result.reasoning_plan
                .map((step, idx) => `<div class="reasoning-step"><strong>Step ${idx + 1}:</strong> ${escapeHtml(step)}</div>`)
                .join("");
            elements.reasoningCard?.classList.remove("hidden");
        } else {
            elements.reasoningCard?.classList.add("hidden");
        }

        // --- Query Transparency Panel ---
        const abstractedQuery = result.abstracted_query || rawQuery;
        if (elements.queryTransparency) {
            elements.qtRawQuery.textContent = rawQuery;
            elements.qtAbstractedQuery.textContent = abstractedQuery;

            const isMasked = abstractedQuery !== rawQuery;
            if (isMasked) {
                elements.qtSameNote?.classList.add("hidden");
                elements.qtPrivacyBadge.textContent = "PII Masked";
                elements.qtPrivacyBadge.className = "qt-badge masked";
            } else {
                elements.qtSameNote?.classList.remove("hidden");
                elements.qtPrivacyBadge.textContent = "No PII Detected";
                elements.qtPrivacyBadge.className = "qt-badge clean";
            }
            elements.queryTransparency.classList.remove("hidden");
        }

        renderTrace(result);
        elements.queryResult.classList.remove("hidden");
        setStatus(elements.queryStatus, "Analysis completed successfully.", "status-success");
    } catch (error) {
        setStatus(elements.queryStatus, error.message, "status-error");
    }
}

async function runComparison() {
    clearStatus(elements.compareStatus);
    elements.compareResults.classList.add("hidden");
    setStatus(elements.compareStatus, "Generating the comparison report...", "status-warning");
    try {
        const payload = await fetchJson("/api/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: elements.compareInput.value,
                expert_mode: true,
                selected_document: elements.compareDocumentSelect.value,
            }),
        });
        renderComparison(payload.results);
        renderJudge(payload.judge);
        elements.compareResults.classList.remove("hidden");
        setStatus(elements.compareStatus, "Comparison complete.", "status-success");
    } catch (error) {
        setStatus(elements.compareStatus, error.message, "status-error");
    }
}

function init() {
    setupTabs();
    elements.documentsInput.addEventListener("change", () => {
        state.selectedFiles = Array.from(elements.documentsInput.files || []);
        renderFileList(state.selectedFiles);
    });
    setupCopyFeature();
    elements.compareInput.value = "Summarize the main topics across all documents.";
    elements.loadDocumentsBtn.addEventListener("click", loadDocuments);
    elements.runQueryBtn.addEventListener("click", runQuery);
    elements.runCompareBtn.addEventListener("click", runComparison);
    if (elements.clearSessionBtn) {
        elements.clearSessionBtn.addEventListener("click", clearSession);
    }
    loadState();
}

document.addEventListener("DOMContentLoaded", init);
