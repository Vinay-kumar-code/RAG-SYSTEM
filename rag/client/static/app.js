const $ = (sel) => document.querySelector(sel);
const statusEl = $('#status');

function setStatus(msg, type = 'info') {
  statusEl.textContent = msg || '';
  statusEl.className = `status ${type}`;
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

async function loadStores() {
  const data = await fetchJSON('/api/stores');
  const sel = $('#store-select');
  sel.innerHTML = '';
  data.stores.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = s.id;
    opt.title = s.embedding_model || '';
    sel.appendChild(opt);
  });
}

async function loadModels() {
  try {
    const data = await fetchJSON('/api/models');
    const sel = $('#model-select');
    sel.innerHTML = '';

    if (data.warning) setStatus(data.warning, 'warn');
    if (data.error) setStatus(data.error, 'error');

    const models = Array.isArray(data.models) ? data.models : [];
    if (models.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No models found (start Ollama and pull models)';
      opt.disabled = true;
      opt.selected = true;
      sel.appendChild(opt);
      return;
    }

    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    });

    // Prefer a common default if present
    const prefer = ['llama3:latest', 'llama3', 'qwen2:7b', 'phi3:mini'];
    const found = prefer.find(p => models.includes(p));
    if (found) sel.value = found;
  } catch (err) {
    setStatus('Failed to load models', 'error');
    showError(err);
  }
}

function addBubble(role, text) {
  const win = $('#chat-window');
  const div = document.createElement('div');
  div.className = `bubble ${role}`;
  div.textContent = text;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

function renderSources(sources) {
  const box = $('#sources');
  box.innerHTML = '';
  (sources || []).forEach((s, idx) => {
    const card = document.createElement('details');
    const title = s.source || s.file || s.path || `Source ${idx+1}`;
    card.innerHTML = `<summary>${title}</summary>`;

    const pre = document.createElement('pre');
    const meta = { ...s };
    delete meta.snippet;
    delete meta.score;

    pre.textContent = `Score: ${s.score}\n\nSnippet:\n${s.snippet}\n\nMeta:\n${JSON.stringify(meta, null, 2)}`;
    card.appendChild(pre);
    box.appendChild(card);
  });
}

function showError(err) {
  const box = $('#error-box');
  const pre = $('#error-text');
  pre.textContent = String(err?.message || err || 'Unknown error');
  box.hidden = false;
}

function hideError() {
  $('#error-box').hidden = true;
}

async function onBuild(e) {
  e.preventDefault();
  hideError();
  setStatus('Building store...', 'info');

  const form = e.currentTarget;
  const data = new FormData(form);

  try {
    const resp = await fetchJSON('/api/build', { method: 'POST', body: data });
    setStatus(`Built store ${resp.store_id} with ${resp.chunks} chunks using ${resp.embedding_model}`, 'success');
    await loadStores();
  } catch (err) {
    setStatus('Build failed', 'error');
    showError(err);
  }
}

async function onSearch() {
  hideError();
  const store = $('#store-select').value;
  const q = $('#search-input').value;
  if (!store || !q) return;
  setStatus('Searching...', 'info');
  try {
    const data = await fetchJSON(`/api/search?store_id=${encodeURIComponent(store)}&q=${encodeURIComponent(q)}`);
    renderSources(data.results.map(r => ({ snippet: r.content.slice(0, 500), score: r.score, ...(r.metadata||{}) })));
    setStatus('Search done', 'success');
  } catch (err) {
    setStatus('Search failed', 'error');
    showError(err);
  }
}

async function onChat(e) {
  e.preventDefault();
  hideError();
  const question = $('#question').value.trim();
  if (!question) return;

  const store = $('#store-select').value;
  const model = $('#model-select').value;
  const k = parseInt($('#topk').value || '4', 10);

  if (!model) {
    setStatus('Pick a model first (Models dropdown)', 'error');
    return;
  }

  addBubble('user', question);
  $('#question').value = '';
  setStatus('Generating...', 'info');

  try {
    const payload = { model, question, k };
    if (store) payload.store_id = store; // omit to allow general chat

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || res.statusText);
    }

    const resp = await res.json();
    addBubble('assistant', resp.answer);
    renderSources(resp.sources);
    setStatus('Done', 'success');
  } catch (err) {
    setStatus('Chat failed', 'error');
    showError(err);
  }
}

function exportChat() {
  const bubbles = Array.from(document.querySelectorAll('#chat-window .bubble'));
  const lines = bubbles.map(b => `${b.classList.contains('user') ? 'User' : 'Assistant'}: ${b.textContent}`);
  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'chat.txt';
  a.click();
  URL.revokeObjectURL(url);
}

window.addEventListener('DOMContentLoaded', async () => {
  $('#build-form').addEventListener('submit', onBuild);
  $('#chat-form').addEventListener('submit', onChat);
  $('#export-btn').addEventListener('click', exportChat);
  $('#search-btn').addEventListener('click', onSearch);

  await Promise.all([loadStores(), loadModels()]);
  setStatus('Ready');
});
