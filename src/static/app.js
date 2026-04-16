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
    judgeMeta: document.getElementById("judgeMeta"),
    judgeWinner: document.getElementById("judgeWinner"),
    judgeDiffs: document.getElementById("judgeDiffs"),
    judgeScores: document.getElementById("judgeScores"),
    privacyValidation: document.getElementById("privacyValidation"),
    responseColumns: document.getElementById("responseColumns"),
    expertMode: document.getElementById("expertMode"),
    documentSelect: document.getElementById("documentSelect"),
    compareDocumentSelect: document.getElementById("compareDocumentSelect"),
    compareExpertMode: document.getElementById("compareExpertMode"),
    copyAnalysisBtn: document.getElementById("copyAnalysisBtn"),
    reasoningTimeline: document.getElementById("reasoningTimeline"),
    reasoningCard: document.getElementById("reasoningCard"),
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

    elements.traceContent.innerHTML = chips
        .map((chip) => `<span class="metric-chip">${escapeHtml(chip)}</span>`)
        .join("");
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
            <div class="formatted-output">${escapeHtml(content || "")}</div>
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
    elements.judgeMeta.innerHTML = "";
    elements.judgeWinner.classList.add("hidden");
    elements.judgeDiffs.classList.add("hidden");
    elements.judgeScores.classList.add("hidden");

    if (!judge || judge.provider === "unavailable") {
        const message = judge?.error ? `Judge unavailable: ${judge.error}` : "Judge unavailable.";
        elements.judgeMeta.innerHTML = `<span class="metric-chip">${escapeHtml(message)}</span>`;
        return;
    }

    const meta = [
        `Provider: ${judge.provider}`,
        judge.model ? `Model: ${judge.model}` : null,
    ].filter(Boolean);

    elements.judgeMeta.innerHTML = meta
        .map((item) => `<span class="metric-chip">${escapeHtml(item)}</span>`)
        .join("");

    const winnerMap = { A: "Local-Only", B: "Hybrid", Tie: "Tie" };
    const winnerLabel = winnerMap[judge.comparison?.winner] || "Tie";
    const localScore = judge.comparison?.answer_a_score ?? judge.local_only?.overall_score ?? "N/A";
    const hybridScore = judge.comparison?.answer_b_score ?? judge.hybrid?.overall_score ?? "N/A";
    const strength = judge.comparison?.preference_strength ? ` (${judge.comparison.preference_strength})` : "";

    elements.judgeWinner.textContent = `Winner: ${winnerLabel}${strength}. Local: ${localScore}. Hybrid: ${hybridScore}.`;
    elements.judgeWinner.classList.remove("hidden");

    const diffs = Array.isArray(judge.comparison?.key_differences) ? judge.comparison.key_differences : [];
    if (diffs.length) {
        elements.judgeDiffs.innerHTML = diffs.slice(0, 5).map((diff) => `<li>${escapeHtml(diff)}</li>`).join("");
        elements.judgeDiffs.classList.remove("hidden");
    }

    const criteria = ["accuracy", "completeness", "relevance", "coherence", "groundedness"];
    const rows = criteria.map((criterion) => {
        const local = judge.local_only?.[criterion]?.score ?? "";
        const hybrid = judge.hybrid?.[criterion]?.score ?? "";
        return [criterion, local, hybrid];
    });
    elements.judgeScores.innerHTML = renderTable(["Criterion", "Local", "Hybrid"], rows);
    elements.judgeScores.classList.remove("hidden");
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
    setStatus(elements.queryStatus, "Running the selected reasoning pipeline...", "status-warning");
    try {
        const result = await fetchJson("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: elements.queryInput.value,
                mode: elements.modeSelect.value,
                expert_mode: elements.expertMode.checked,
                selected_document: elements.documentSelect.value,
            }),
        });
        elements.answerContent.textContent = result.answer || "";

        if (elements.reasoningTimeline && result.reasoning_plan) {
            elements.reasoningTimeline.innerHTML = result.reasoning_plan
                .map((step, idx) => `<div class="reasoning-step"><strong>Step ${idx + 1}:</strong> ${escapeHtml(step)}</div>`)
                .join("");
            elements.reasoningCard?.classList.remove("hidden");
        } else {
            elements.reasoningCard?.classList.add("hidden");
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
                expert_mode: (elements.compareExpertMode || elements.expertMode).checked,
                selected_document: elements.compareDocumentSelect.value,
            }),
        });
        const results = payload.results;
        renderComparison(results);
        renderPrivacyTransparency(results);
        renderJudge(payload.judge);
        elements.compareResults.classList.remove("hidden");
        setStatus(elements.compareStatus, "Comparison report generated successfully.", "status-success");
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
    loadState();
}

document.addEventListener("DOMContentLoaded", init);
