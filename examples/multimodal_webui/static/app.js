'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {models: [], sessions: [], current: null, attachments: [], busy: false, uploading: 0, controller: null, thinking: false};
let toastTimer;
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5000); }
function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { detail = response.statusText; }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return response;
}
function jsonOptions(body) { return {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}; }
function session() { return state.sessions.find(s => s.id === state.current); }
function save() {
  try { localStorage.setItem('omni.sessions.v1', JSON.stringify(state.sessions.slice(0, 30))); }
  catch { toast('浏览器存储已满；可导出消息后清理旧对话。'); }
  renderHistory();
}
function newChat() {
  if (state.busy || state.uploading) return;
  const item = {id: crypto.randomUUID(), title: '新的探索', messages: [], model: $('modelSelect').value, created: Date.now()};
  state.sessions.unshift(item); state.sessions = state.sessions.slice(0, 30); state.current = item.id;
  state.attachments = []; $('prompt').value = ''; notice(''); renderAttachments(); renderMessages(); save(); $('prompt').focus(); $('sidebar').classList.remove('open');
}
function renderHistory() {
  const items = state.sessions.filter(s => s.messages.length);
  $('history').innerHTML = items.length ? items.map(s => `<button class="history-item ${s.id === state.current ? 'active' : ''}" data-session="${escapeHTML(s.id)}"><span>${escapeHTML(s.title)}</span><small>↗</small></button>`).join('') : '<p class="history-empty">每一个好问题，<br>都是一次新的探索。</p>';
  $('history').querySelectorAll('button').forEach(button => button.onclick = () => {
    if (state.busy || state.uploading) return toast('请先停止当前生成或等待上传完成');
    state.current = button.dataset.session; state.attachments = []; $('prompt').value = ''; renderAttachments();
    const item = session();
    if (state.models.some(m => m.id === item.model)) $('modelSelect').value = item.model;
    renderMessages(); renderHistory(); updateModel(); $('sidebar').classList.remove('open');
  });
}
function renderModelList(data) {
  const previous = $('modelSelect').value || session()?.model;
  state.models = data.models;
  $('modelSelect').innerHTML = data.models.length ? data.models.map(m => `<option value="${escapeHTML(m.id)}" ${m.online ? '' : 'disabled'}>${escapeHTML(m.name)} · ${new URL(m.url).port}${m.online ? '' : ' · 离线'}</option>`).join('') : '<option value="">未发现模型 · 添加本地服务</option>';
  if (state.models.some(m => m.id === previous)) $('modelSelect').value = previous;
  else if (state.models.some(m => m.online)) $('modelSelect').value = state.models.find(m => m.online).id;
  $('serviceCount').textContent = state.models.filter(m => m.online).length;
  $('serviceList').innerHTML = state.models.length ? state.models.map(m => `<div class="service-row"><i class="dot ${m.online ? '' : 'offline'}"></i><div><strong>${escapeHTML(m.name)}</strong><small>${escapeHTML(m.url)} · ${m.adapter === 'beebee' ? 'BeeBee 原生' : 'OpenAI 兼容'}</small></div><span class="service-caps">${m.online ? '在线' : '离线'}<br>${escapeHTML(m.capabilities.map(c => ({text:'文本',image:'图像',video:'视频',audio:'音频'}[c] || c)).join(' / '))}${m.capabilities_known ? '' : ' · 待确认'}</span></div>`).join('') : '<div class="empty-services">还没有发现本地模型服务。<br>启动 SGLang / vLLM 后会自动出现在这里，也可以在下方添加地址。</div>';
  if (data.error) toast(data.error);
  updateModel();
}
async function refreshModels(force = false) {
  try { renderModelList(await (await api(force ? '/api/discover' : '/api/models', force ? {method:'POST'} : {})).json()); }
  catch (error) { $('connectionLabel').textContent = '中间层连接中断'; if (force) toast(error.message); }
}
function updateModel() {
  const model = state.models.find(m => m.id === $('modelSelect').value);
  $('modelDot').classList.toggle('offline', !model?.online);
  $('connectionLabel').textContent = model?.online ? '本地连接 · 自动发现' : '等待本地模型服务';
  $('downsample').disabled = state.busy;
  $('send').disabled = !state.busy && (!model?.online || state.uploading > 0);
  if (session() && !state.busy) { session().model = $('modelSelect').value; }
  updateSettings();
}
function settings() { return {thinking: state.thinking, downsample: Number($('downsample').value), video_layout: $('videoLayout').value, video_fps: Number($('videoFps').value), max_tokens: Number($('maxTokens').value), temperature: Number($('temperature').value), top_p: Number($('topP').value), presence_penalty: Number($('penalty').value), video_frames: Number($('videoFrames').value), max_edge: Number($('maxEdge').value), video_audio: $('videoAudio').checked, system: $('systemPrompt').value}; }
function updateSettings() {
  $('thinkingToggle').setAttribute('aria-pressed', String(state.thinking));
  $('temperatureValue').textContent = $('temperature').value; $('topPValue').textContent = $('topP').value; $('penaltyValue').textContent = $('penalty').value;
  const model = state.models.find(m => m.id === $('modelSelect').value);
  $('settingsSummary').textContent = `${state.thinking ? 'Thinking' : 'Non-thinking'} · ${$('maxTokens').value} tokens${model?.adapter === 'beebee' ? ' · Down ' + $('downsample').value : ''}`;
}
function attachmentHTML(file, removable) {
  return `<div class="attachment">${file.kind === 'image' ? `<img src="/api/media/${escapeHTML(file.id)}" alt="图片附件">` : `<span class="file-icon">${file.kind === 'audio' ? '≋' : '▷'}</span>`}<div><div class="file-name">${escapeHTML(file.name)}</div><small>${(file.size / 1024 / 1024).toFixed(1)} MB${file.duration ? ' · ' + Math.round(file.duration) + 's' : ''}</small></div>${removable ? `<button class="remove" data-remove="${escapeHTML(file.id)}" aria-label="移除附件">×</button>` : ''}</div>`;
}
function renderAttachments() {
  $('attachments').innerHTML = state.attachments.map(f => attachmentHTML(f, true)).join('');
  $('attachments').querySelectorAll('[data-remove]').forEach(b => b.onclick = () => { state.attachments = state.attachments.filter(f => f.id !== b.dataset.remove); renderAttachments(); });
}
async function uploadFiles(files) {
  if (state.busy || state.uploading) return toast('请等待当前操作完成');
  if (state.attachments.length + files.length > 8) return toast('每条消息最多添加 8 个附件');
  state.uploading++; updateModel(); $('attach').disabled = true; notice('正在上传并检查媒体…');
  try {
    for (const file of files) {
      if (file.size > 100 * 1024 * 1024) throw new Error(`${file.name} 超过 100 MB`);
      const body = new FormData(); body.append('file', file);
      state.attachments.push(await (await api('/api/upload', {method:'POST', body})).json()); renderAttachments();
    }
    notice('');
  } catch (error) { notice(error.message); }
  finally { state.uploading--; $('attach').disabled = false; $('fileInput').value = ''; updateModel(); }
}
function inlineMarkdown(text) {
  return escapeHTML(text).replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
}
function markdown(text) {
  // Escape before rendering. Never insert model-provided HTML into the page.
  return String(text).split(/```/).map((part, i) => {
    if (i % 2) return `<pre><code>${escapeHTML(part.replace(/^[\w+-]*\n/, ''))}</code></pre>`;
    return part.split(/\n\n+/).filter(Boolean).map(block => {
      const rows = block.trim().split('\n');
      if (rows.length > 1 && /^\|?.*\|/.test(rows[0]) && /^[\s|:-]+$/.test(rows[1])) {
        const cells = row => row.replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => inlineMarkdown(c.trim()));
        return `<table><thead><tr>${cells(rows[0]).map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.slice(2).map(r => `<tr>${cells(r).map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
      }
      return `<p>${inlineMarkdown(block.replace(/^#{1,6} /gm, ''))}</p>`;
    }).join('');
  }).join('');
}
function mediaHTML(files) {
  return (files || []).map(file => {
    const url = `/api/media/${escapeHTML(file.id)}`;
    if (file.kind === 'image') return `<a href="${url}" target="_blank" rel="noopener"><img src="${url}" alt="${escapeHTML(file.name)}" loading="lazy"></a>`;
    return `<div>${file.kind === 'video' ? `<video src="${url}" controls preload="metadata"></video>` : `<audio src="${url}" controls preload="metadata"></audio>`}</div>`;
  }).join('');
}
function messageHTML(message, index) {
  const assistant = message.role === 'assistant';
  let reasoning = '';
  if (message.reasoning || message.reasoning_blocks?.length) {
    const blocks = message.reasoning_blocks || [];
    const emptyInitial = blocks.length && blocks[0].complete && !blocks[0].text.trim();
    const label = emptyInitial ? '首段推理为空' : '思考过程';
    const reopened = blocks.length > 1 ? ` · 另有 ${blocks.length - 1} 段推理` : '';
    reasoning = `<details class="reasoning"><summary>✧ ${label}${reopened}</summary><div class="reasoning-text">${escapeHTML(message.reasoning || (emptyInitial ? '模型原生 thinking 首段为空，未补写推理。' : '模型尚未输出推理文本。'))}</div></details>`;
  }
  const usage = message.usage || {};
  const meta = [usage.completion_tokens !== undefined ? `${usage.completion_tokens} tokens` : '', message.elapsed ? `${message.elapsed.toFixed(1)}s` : '', message.finish_reason === 'length' ? '达到输出长度上限' : '', message.stopped ? '已停止' : ''].filter(Boolean).join(' · ');
  return `<article class="message ${message.role}" data-index="${index}"><div class="message-head"><span class="message-avatar">${assistant ? '✳' : 'Y'}</span><span>${assistant ? escapeHTML(message.modelName || 'Omni') : '你'}</span></div>${message.attachments?.length ? `<div class="message-media">${mediaHTML(message.attachments)}</div>` : ''}${reasoning}<div class="message-content">${assistant ? markdown(message.content || '') : escapeHTML(message.content)}</div>${message.pending && !message.content && !message.reasoning ? `<div class="status-text">${escapeHTML(message.status || '等待模型回复…')}</div>` : ''}${message.error ? `<div class="error-text">${escapeHTML(message.error)}</div>` : ''}${assistant ? `<div class="message-actions"><button data-copy="${index}">复制</button><button data-export="${index}">原始输出 ↗</button>${!message.pending ? `<button data-retry="${index}">重新生成</button>` : ''}<span>${escapeHTML(meta)}</span></div>` : ''}</article>`;
}
function bindMessageActions(container) {
  container.querySelectorAll('[data-copy]').forEach(b => b.onclick = async () => {
    try { await navigator.clipboard.writeText(session().messages[Number(b.dataset.copy)].content); toast('已复制回复'); }
    catch { toast('当前浏览器不支持复制，请选中文本复制'); }
  });
  container.querySelectorAll('[data-export]').forEach(b => b.onclick = () => {
    const message = session().messages[Number(b.dataset.export)];
    const url = URL.createObjectURL(new Blob([JSON.stringify(message, null, 2)], {type:'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = 'omni-response.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  container.querySelectorAll('[data-retry]').forEach(b => b.onclick = () => {
    if (state.busy || state.uploading) return;
    if (!state.models.some(m => m.id === $('modelSelect').value && m.online)) return toast('请先选择在线模型');
    const index = Number(b.dataset.retry);
    if (index !== session().messages.length - 1) return toast('可重新生成最后一条回复；早期回复请新建对话');
    session().messages.splice(index); generate();
  });
}
function renderMessages() {
  const messages = session()?.messages || [];
  $('welcome').hidden = !!messages.length;
  $('messages').innerHTML = messages.map(messageHTML).join(''); bindMessageActions($('messages'));
  $('chatScroll').scrollTop = $('chatScroll').scrollHeight;
}
function updateReply(reply) {
  const scroll = $('chatScroll'); const pinned = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 150;
  const last = $('messages').lastElementChild;
  const open = last?.querySelector('.reasoning')?.open;
  if (last) { last.outerHTML = messageHTML(reply, session().messages.length - 1); const replacement = $('messages').lastElementChild; if (open && replacement.querySelector('.reasoning')) replacement.querySelector('.reasoning').open = true; bindMessageActions(replacement); }
  if (pinned) scroll.scrollTop = scroll.scrollHeight;
}
function setBusy(busy) {
  state.busy = busy;
  for (const id of ['modelSelect','newChat','attach','thinkingToggle','maxTokens','temperature','topP','penalty','downsample','videoFrames','videoLayout','videoFps','maxEdge','videoAudio','systemPrompt']) $(id).disabled = busy;
  $('send').textContent = busy ? '■' : '↑'; $('send').classList.toggle('generating', busy); $('send').setAttribute('aria-label', busy ? '停止生成' : '发送消息');
  updateModel();
}
async function generate() {
  const model = state.models.find(m => m.id === $('modelSelect').value);
  if (!model?.online) return toast('请先选择在线模型');
  const item = session();
  const request = {model_id: model.id, settings: settings(), messages: item.messages.map(m => ({role:m.role, content:m.content, attachments: (m.attachments || []).map(f => f.id)}))};
  const reply = {role:'assistant', content:'', raw:'', modelName:model.name, modelUrl:model.url, settings:request.settings, pending:true};
  item.messages.push(reply); item.model = model.id; state.controller = new AbortController(); setBusy(true); renderMessages(); notice('');
  const started = performance.now();
  try {
    const response = await api('/api/chat', {...jsonOptions(request), signal:state.controller.signal});
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ''; let completed = false;
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      buffer += decoder.decode(value, {stream:true}); buffer = buffer.replace(/\r\n/g, '\n');
      let end;
      while ((end = buffer.indexOf('\n\n')) !== -1) {
        const event = buffer.slice(0, end); buffer = buffer.slice(end + 2);
        const data = event.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
        if (!data) continue;
        const payload = JSON.parse(data);
        if (payload.type === 'error') throw new Error(payload.message);
        if (payload.type === 'status') reply.status = payload.message;
        if (payload.type === 'snapshot') Object.assign(reply, payload);
        if (payload.type === 'done') { reply.usage = payload.usage; reply.finish_reason = payload.finish_reason; completed = true; }
        updateReply(reply);
      }
    }
    if (!completed) throw new Error('连接提前结束；已保留收到的内容，可重试。');
  } catch (error) {
    if (error.name === 'AbortError') reply.stopped = true;
    else { reply.error = error.message; notice(error.message); }
  } finally {
    reply.pending = false; reply.elapsed = (performance.now() - started) / 1000;
    state.controller?.abort();
    state.controller = null; setBusy(false); updateReply(reply); save();
  }
}
async function send() {
  if (state.busy) return state.controller?.abort();
  if (state.uploading) return toast('媒体还在上传');
  let content = $('prompt').value.trim();
  if (!content && !state.attachments.length) return $('prompt').focus();
  const model = state.models.find(m => m.id === $('modelSelect').value);
  if (!model?.online) return openServices();
  if (!session()) newChat();
  const item = session();
  // Failed/empty responses must be retried rather than sent as false history.
  const last = item.messages.at(-1);
  if (last?.role === 'user') return toast('上一条问题尚未回复，请先重新发送或新建对话');
  if (last?.error || (last?.role === 'assistant' && !last.content)) return toast('上一条回复没有完成，请点击“重新生成”或新建对话');
  if (!content) content = '请描述并分析这些媒体内容。';
  const attachments = [...state.attachments];
  item.messages.push({role:'user', content, attachments});
  if (item.messages.length === 1) item.title = content.slice(0, 26);
  state.attachments = []; $('prompt').value = ''; $('prompt').style.height = ''; renderAttachments(); await generate();
}
function openServices() { $('servicesDialog').showModal(); refreshModels(true); }
function toggleSettings(show) { $('settingsPanel').hidden = !show; $('settingsToggle').setAttribute('aria-expanded', String(show)); }
$('newChat').onclick = newChat; $('chatNav').onclick = () => $('sidebar').classList.remove('open');
$('servicesNav').onclick = openServices; $('closeServices').onclick = () => $('servicesDialog').close();
$('servicesDialog').onclick = event => { if (event.target === $('servicesDialog') && event.offsetX >= $('servicesDialog').clientWidth) $('servicesDialog').close(); };
$('refreshModels').onclick = () => refreshModels(true); $('modelSelect').onchange = () => { updateModel(); save(); };
$('settingsToggle').onclick = () => toggleSettings($('settingsPanel').hidden); $('closeSettings').onclick = () => toggleSettings(false);
$('menuToggle').onclick = () => $('sidebar').classList.toggle('open');
document.addEventListener('click', event => { if (!$('sidebar').contains(event.target) && !$('menuToggle').contains(event.target)) $('sidebar').classList.remove('open'); });
$('clearHistory').onclick = () => { if (state.busy || state.uploading) return; if (confirm('清空此浏览器保存的全部对话记录？')) { state.sessions = []; newChat(); } };
$('thinkingToggle').onclick = () => { state.thinking = !state.thinking; updateSettings(); };
for (const id of ['maxTokens','temperature','topP','penalty','videoFrames','videoLayout','videoFps','maxEdge','videoAudio','systemPrompt']) $(id).oninput = updateSettings;
$('downsample').onchange = updateSettings;
$('attach').onclick = () => { $('fileInput').accept = '.jpg,.jpeg,.png,.webp,.gif,.bmp,.mp4,.mov,.webm,.mkv,.wav,.mp3,.m4a,.flac,.ogg'; $('fileInput').click(); };
$('fileInput').onchange = event => uploadFiles([...event.target.files]);
document.querySelectorAll('.suggestion').forEach(button => button.onclick = () => {
  const kind = button.dataset.kind;
  $('prompt').value = {image:'请仔细观察这张图片，描述画面内容并提取其中的文字。',video:'请按时间顺序总结这段视频的内容，并指出关键时刻。',audio:'请转录这段音频，并总结主要内容。'}[kind];
  $('fileInput').accept = {image:'.jpg,.jpeg,.png,.webp,.gif,.bmp',video:'.mp4,.mov,.webm,.mkv',audio:'.wav,.mp3,.m4a,.flac,.ogg'}[kind]; $('fileInput').click();
});
for (const name of ['dragenter','dragover']) $('dropZone').addEventListener(name, event => { event.preventDefault(); $('dropZone').classList.add('dragover'); });
$('dropZone').addEventListener('dragleave', () => $('dropZone').classList.remove('dragover'));
$('dropZone').addEventListener('drop', event => { event.preventDefault(); $('dropZone').classList.remove('dragover'); uploadFiles([...event.dataTransfer.files]); });
$('prompt').addEventListener('paste', event => { if (event.clipboardData.files.length) { event.preventDefault(); uploadFiles([...event.clipboardData.files]); } });
$('prompt').addEventListener('input', () => { $('prompt').style.height = ''; $('prompt').style.height = Math.min($('prompt').scrollHeight, 180) + 'px'; });
$('prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); } });
document.addEventListener('keydown', event => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); newChat(); } });
$('send').onclick = send;
$('addServiceForm').onsubmit = async event => {
  event.preventDefault(); const button = event.target.querySelector('button'); button.disabled = true;
  try { renderModelList(await (await api('/api/services', jsonOptions({url:$('serviceUrl').value, name:$('serviceName').value}))).json()); toast('服务地址已保存，模型列表已更新'); }
  catch (error) { toast(error.message); }
  finally { button.disabled = false; }
};
try { state.sessions = JSON.parse(localStorage.getItem('omni.sessions.v1') || '[]'); if (!Array.isArray(state.sessions)) state.sessions = []; state.sessions.forEach(s => s.messages.forEach(m => { if (m.pending) { m.pending = false; m.stopped = true; } })); } catch { state.sessions = []; }
newChat(); refreshModels(true); setInterval(() => refreshModels(), 10000);
