// CrowdVision Command Center - Client Logic
let trendChart = null;
let showBoxes = true;
let showPoints = true;
let showZones = true;
let showHeatmap = false;
let currentCrowdMode = "auto";
let currentEngine = "yolo";
let selectedFile = null;
let isRecording = false;
let isEmergency = false;
let ws = null;

// Zone Management State
let activeZones = [];
let isDrawingMode = false;
let drawnPoints = []; // [{x, y}] normalized 0.0 - 1.0
let canvas = null;
let ctx = null;

// Trend history arrays (max 15 data points)
const historyLabels = [];
const historyTotal = [];
let dynamicZoneHistories = {}; // zone_id -> array of counts

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide Icons
  if (window.lucide) {
    lucide.createIcons();
  }

  // Start live clock
  initClock();

  // Initialize Canvas for Drawing
  initDrawingCanvas();

  // Initialize Chart.js
  initTrendChart();

  // Connect to video stream
  startStream();

  // Connect telemetry WebSocket
  connectWebSocket();

  // Fetch initial video & zones list
  fetchVideosList();
  fetchZonesList();

  // Setup drag & drop
  setupDropzone();
});

// CLOCK
function initClock() {
  function update() {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0] + ' WIB';
    const clockEl = document.getElementById('live-clock');
    const hudTimeEl = document.getElementById('hud-timestamp');
    if (clockEl) clockEl.textContent = timeStr;
    if (hudTimeEl) hudTimeEl.textContent = `${now.getDate()} ${now.toLocaleString('default', { month: 'short' })} ${now.getFullYear()} ${timeStr}`;
  }
  update();
  setInterval(update, 1000);
}

// VIDEO STREAM
function startStream() {
  const img = document.getElementById('live-stream-img');
  const loader = document.getElementById('stream-loading');
  img.src = '/api/stream?t=' + Date.now();
  img.style.display = 'block';
  if (loader) loader.style.display = 'none';
}

function handleStreamLoaded() {
  const loader = document.getElementById('stream-loading');
  if (loader) loader.style.display = 'none';
  const canvasEl = document.getElementById('zone-drawing-canvas');
  const img = document.getElementById('live-stream-img');
  const vp = document.getElementById('video-viewport');
  if (canvasEl && img && vp && img.clientWidth > 0) {
    const rect = img.getBoundingClientRect();
    const vpRect = vp.getBoundingClientRect();
    canvasEl.style.top = `${rect.top - vpRect.top}px`;
    canvasEl.style.left = `${rect.left - vpRect.left}px`;
    canvasEl.style.width = `${rect.width}px`;
    canvasEl.style.height = `${rect.height}px`;
    canvasEl.width = Math.round(rect.width);
    canvasEl.height = Math.round(rect.height);
  }
}

function handleStreamError() {
  const loader = document.getElementById('stream-loading');
  if (loader) loader.style.display = 'flex';
  setTimeout(() => {
    const img = document.getElementById('live-stream-img');
    img.src = '/api/stream?t=' + Date.now();
  }, 3000);
}

// WEBSOCKET & TELEMETRY
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const stats = JSON.parse(event.data);
      updateDashboardStats(stats);
    } catch (e) {
      console.error("WS Parse error", e);
    }
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = () => {
    fallbackPollStats();
  };
}

let pollingInterval = null;
function fallbackPollStats() {
  if (pollingInterval) return;
  pollingInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const stats = await res.json();
        updateDashboardStats(stats);
      }
    } catch (e) {}
  }, 1000);
}

// DASHBOARD STATS UPDATER (DYNAMIC ZONES)
function updateDashboardStats(stats) {
  if (!stats) return;

  const total = stats.total || 0;
  const totalAll = stats.total_detected_all || total;
  const zones = stats.zones || [];
  const zonesCount = stats.zones_count !== undefined ? stats.zones_count : zones.length;

  // Total People in Zone
  document.getElementById('stat-total-people').textContent = total.toLocaleString();

  // Subtitle info
  const subInfoEl = document.getElementById('stat-sub-info');
  if (subInfoEl) {
    if (zonesCount > 0) {
      subInfoEl.innerHTML = `Menghitung orang di dalam <strong>${zonesCount} zona aktif</strong> (Total terdeteksi di frame: ${totalAll})`;
    } else {
      subInfoEl.innerHTML = `<span class="text-yellow">Belum ada zona. Klik '+ Buat Zona' untuk mulai menghitung per area.</span>`;
    }
  }

  // Update badge counts
  const countBadge = document.getElementById('zones-count-badge');
  const sidebarCountBadge = document.getElementById('sidebar-zone-count');
  if (countBadge) countBadge.textContent = zonesCount;
  if (sidebarCountBadge) sidebarCountBadge.textContent = zonesCount;

  // Render Dynamic Zone Cards in Bottom Panel
  renderDynamicZoneCards(zones, total);

  // FPS & Latency
  if (stats.fps) {
    document.getElementById('fps-val').textContent = Math.round(stats.fps);
  }
  if (stats.inference_ms) {
    document.getElementById('latency-val').textContent = `${Math.round(stats.inference_ms)}ms`;
  }

  // Active Video Name & Model
  if (stats.video_name && stats.video_name !== "No Video Active") {
    const titleEl = document.getElementById('feed-title');
    if (titleEl) {
      const fullTitle = `LIVE FEED: ${stats.video_name.toUpperCase()}`;
      titleEl.textContent = fullTitle;
      titleEl.title = `Nama File Lengkap: ${stats.video_name}`;
    }
  }

  if (stats.model_name) {
    const modelEl = document.getElementById('model-val');
    if (modelEl) {
      const mName = stats.model_name.includes('/') ? stats.model_name.split('/').pop() : stats.model_name;
      const engStr = (stats.engine || 'yolo').toUpperCase();
      modelEl.textContent = `${engStr}: ${mName}`;
    }
  }

  if (stats.device_name) {
    const devEl = document.getElementById('device-val');
    if (devEl) {
      devEl.textContent = stats.device_name;
      devEl.className = stats.device && stats.device.includes('cuda') ? 'text-green' : (stats.device === 'mps' ? 'text-cyan' : 'text-yellow');
    }
  }

  // Alert check
  if (stats.alert_triggered) {
    triggerHighDensityAlert(zones, total);
  }

  // Update Crowd Eye Telemetry
  if (stats.crowd_eye) {
    const eye = stats.crowd_eye;
    const method = (eye.method || 'detection').toUpperCase();
    const ratio = eye.ratio !== undefined ? eye.ratio : 1.0;
    const packedness = eye.packedness !== undefined ? eye.packedness : 0.0;

    const methodValEl = document.getElementById('crowdeye-method-val');
    if (methodValEl) {
      methodValEl.textContent = stats.engine === 'p2pnet' ? 'P2PNET' : method;
      methodValEl.className = stats.engine === 'p2pnet' ? 'text-green' : (method === 'DETECTION' ? 'text-cyan' : 'text-orange');
    }

    const badgeRouteEl = document.getElementById('ce-badge-route');
    if (badgeRouteEl) {
      const badgeText = stats.engine === 'p2pnet' ? 'P2PNET DOTS' : method;
      badgeRouteEl.textContent = badgeText;
      badgeRouteEl.className = `badge-route ${stats.engine === 'p2pnet' ? 'badge-p2p' : (method === 'DETECTION' ? 'badge-detection' : 'badge-density')}`;
    }

    const detCountEl = document.getElementById('ce-det-count');
    if (detCountEl) detCountEl.textContent = (eye.detection_count || 0).toLocaleString();

    const p2pCountEl = document.getElementById('ce-p2p-count');
    if (p2pCountEl) {
      const p2pCnt = eye.p2p_count !== undefined ? eye.p2p_count : (stats.engine === 'p2pnet' ? total : 0);
      p2pCountEl.textContent = p2pCnt.toLocaleString();
    }

    const densCountEl = document.getElementById('ce-dens-count');
    if (densCountEl) densCountEl.textContent = (eye.density_count || 0).toLocaleString();

    const ratioValEl = document.getElementById('ce-ratio-val');
    if (ratioValEl) ratioValEl.innerHTML = `${ratio.toFixed(2)} <span class="text-dim">/ 5.0</span>`;

    const ratioFillEl = document.getElementById('ce-ratio-fill');
    if (ratioFillEl) {
      const ratioPct = Math.min(100, Math.round((ratio / 5.0) * 100));
      ratioFillEl.style.width = `${ratioPct}%`;
      ratioFillEl.style.background = ratio >= 5.0 ? 'linear-gradient(90deg, #ff9933, #ff334b)' : 'linear-gradient(90deg, #00e5ff, #00ff88)';
    }

    const packedValEl = document.getElementById('ce-packed-val');
    if (packedValEl) packedValEl.textContent = packedness.toFixed(3);

    const packedFillEl = document.getElementById('ce-packed-fill');
    if (packedFillEl) {
      const packedPct = Math.min(100, Math.round(packedness * 100));
      packedFillEl.style.width = `${packedPct}%`;
    }

    const scanMsEl = document.getElementById('ce-scan-ms');
    if (scanMsEl && eye.scan_ms) scanMsEl.textContent = Math.round(eye.scan_ms);
  }

  // Update chart with dynamic zones
  appendChartData(total, zones);
}

// DYNAMIC ZONE CARDS RENDERER
function renderDynamicZoneCards(zones, total) {
  const container = document.getElementById('zones-breakdown-container');
  if (!container) return;

  if (!zones || zones.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 10px; text-align: center; color: var(--text-muted); font-size: 11px;">
        Tidak ada zona aktif. <button class="link-btn" onclick="toggleDrawZoneMode()">+ Buat Zona Baru</button> atau <button class="link-btn" onclick="resetDefaultZones()">Reset Default</button>
      </div>
    `;
    return;
  }

  const maxVal = Math.max(1, total);
  let html = '';

  zones.forEach(z => {
    const pct = Math.min(100, Math.round((z.count / maxVal) * 100));
    const hex = z.hex_color || '#00e5ff';

    html += `
      <div class="zone-dynamic-card" style="border-left: 3px solid ${hex};">
        <div class="zone-dyn-header">
          <span class="zone-dyn-title" style="color: ${hex};" title="${z.name}">${z.name}</span>
          <button class="zone-del-btn" onclick="deleteZone('${z.id}')" title="Hapus ${z.name}">
            <i data-lucide="trash" style="width:11px; height:11px;"></i>
          </button>
        </div>
        <div class="zone-dyn-count">${z.count.toLocaleString()}</div>
        <div class="z-progress-bar">
          <div class="z-bar-fill" style="width: ${pct}%; background: ${hex};"></div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
  if (window.lucide) lucide.createIcons();
}

// INTERACTIVE ZONE DRAWING SYSTEM
function initDrawingCanvas() {
  canvas = document.getElementById('zone-drawing-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');

  function resizeCanvas() {
    const vp = document.getElementById('video-viewport');
    const img = document.getElementById('live-stream-img');
    if (!canvas || !vp) return;

    if (img && img.clientWidth > 0 && img.clientHeight > 0) {
      const rect = img.getBoundingClientRect();
      const vpRect = vp.getBoundingClientRect();
      canvas.style.top = `${rect.top - vpRect.top}px`;
      canvas.style.left = `${rect.left - vpRect.left}px`;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
    } else {
      canvas.style.top = '0px';
      canvas.style.left = '0px';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.width = vp.clientWidth;
      canvas.height = vp.clientHeight;
    }
    redrawDrawingPreview();
  }

  window.addEventListener('resize', resizeCanvas);
  const imgEl = document.getElementById('live-stream-img');
  if (imgEl) {
    imgEl.addEventListener('load', resizeCanvas);
  }
  resizeCanvas();

  // Canvas Click to add vertex
  canvas.addEventListener('click', (e) => {
    if (!isDrawingMode) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Save normalized coordinates (0.0 - 1.0)
    const normX = Math.round((x / canvas.width) * 1000) / 1000;
    const normY = Math.round((y / canvas.height) * 1000) / 1000;

    drawnPoints.push([normX, normY]);
    document.getElementById('draw-pts-counter').textContent = `${drawnPoints.length} Titik`;

    redrawDrawingPreview();
  });
}

function toggleDrawZoneMode() {
  isDrawingMode = !isDrawingMode;
  const canvasEl = document.getElementById('zone-drawing-canvas');
  const hudBar = document.getElementById('drawing-hud-bar');

  if (isDrawingMode) {
    drawnPoints = [];
    const vp = document.getElementById('video-viewport');
    const img = document.getElementById('live-stream-img');
    if (canvasEl && img && vp && img.clientWidth > 0) {
      const rect = img.getBoundingClientRect();
      const vpRect = vp.getBoundingClientRect();
      canvasEl.style.top = `${rect.top - vpRect.top}px`;
      canvasEl.style.left = `${rect.left - vpRect.left}px`;
      canvasEl.style.width = `${rect.width}px`;
      canvasEl.style.height = `${rect.height}px`;
      canvasEl.width = Math.round(rect.width);
      canvasEl.height = Math.round(rect.height);
    }
    canvasEl.classList.add('active');
    hudBar.style.display = 'flex';
    document.getElementById('draw-pts-counter').textContent = `0 Titik`;
    redrawDrawingPreview();
    openToast('Mode Gambar Zona Aktif: Klik pada video untuk menentukan titik sudut zona!');
  } else {
    cancelDrawingZone();
  }
  if (window.lucide) lucide.createIcons();
}

function cancelDrawingZone() {
  isDrawingMode = false;
  drawnPoints = [];
  const canvasEl = document.getElementById('zone-drawing-canvas');
  const hudBar = document.getElementById('drawing-hud-bar');
  canvasEl.classList.remove('active');
  hudBar.style.display = 'none';
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  openToast('Pembuatan zona dibatalkan.');
}

function finishDrawingZone() {
  if (drawnPoints.length < 3) {
    openToast('Zona harus memiliki minimal 3 titik sudut!');
    return;
  }
  // Open Save Zone modal to name it & pick color
  openSaveZoneModal();
}

function redrawDrawingPreview() {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (drawnPoints.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;

  ctx.beginPath();
  const [firstX, firstY] = drawnPoints[0];
  ctx.moveTo(firstX * w, firstY * h);

  for (let i = 1; i < drawnPoints.length; i++) {
    const [px, py] = drawnPoints[i];
    ctx.lineTo(px * w, py * h);
  }

  if (drawnPoints.length >= 3) {
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 229, 255, 0.25)';
    ctx.fill();
  }

  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw node circles
  drawnPoints.forEach(([px, py], idx) => {
    ctx.beginPath();
    ctx.arc(px * w, py * h, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#00ff88';
    ctx.fill();
    ctx.strokeStyle = '#060a10';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Node number
    ctx.font = '10px Chakra Petch';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`${idx + 1}`, px * w + 8, py * h - 4);
  });
}

function openSaveZoneModal() {
  document.getElementById('new-zone-name').value = `Zona ${activeZones.length + 1}`;
  document.getElementById('save-zone-modal').classList.add('open');
}

function closeSaveZoneModal() {
  document.getElementById('save-zone-modal').classList.remove('open');
}

async function submitNewZone() {
  const name = document.getElementById('new-zone-name').value.trim() || `Zona Baru`;
  const colorRadio = document.querySelector('input[name="zone-color"]:checked');
  const color = colorRadio ? colorRadio.value : '#00e5ff';

  try {
    const res = await fetch('/api/zones', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        points: drawnPoints,
        hex_color: color
      })
    });

    if (res.ok) {
      openToast(`Zona '${name}' berhasil dibuat dan mulai dipantau!`);
      closeSaveZoneModal();
      cancelDrawingZone();
      fetchZonesList();
    } else {
      openToast('Gagal menyimpan zona baru.');
    }
  } catch (e) {
    openToast('Terjadi kesalahan jaringan.');
  }
}

// ZONE MANAGER MODAL
function openZoneManagerModal() {
  document.getElementById('zone-manager-modal').classList.add('open');
  fetchZonesList();
}

function closeZoneManagerModal() {
  document.getElementById('zone-manager-modal').classList.remove('open');
}

async function fetchZonesList() {
  try {
    const res = await fetch('/api/zones');
    if (!res.ok) return;
    const data = await res.json();
    activeZones = data.zones || [];

    // Render table inside Zone Manager modal
    renderZonesManagerTable();
  } catch (e) {}
}

function renderZonesManagerTable() {
  const tableEl = document.getElementById('zones-table-list');
  if (!tableEl) return;

  if (activeZones.length === 0) {
    tableEl.innerHTML = '<div style="padding:15px; text-align:center; color:var(--text-muted); font-size:11px;">Belum ada zona deteksi yang dibuat.</div>';
    return;
  }

  tableEl.innerHTML = activeZones.map(z => `
    <div class="zone-table-row">
      <div class="zone-meta-left">
        <span class="zone-color-indicator" style="background: ${z.hex_color};"></span>
        <div>
          <div class="zone-name-text">${z.name}</div>
          <div class="zone-pts-tag">${z.points ? z.points.length : 0} titik sudut</div>
        </div>
      </div>
      <div class="zone-row-actions">
        <button class="btn-trash-zone" onclick="deleteZone('${z.id}')">
          <i data-lucide="trash-2"></i> Hapus
        </button>
      </div>
    </div>
  `).join('');

  if (window.lucide) lucide.createIcons();
}

async function deleteZone(zoneId) {
  try {
    const res = await fetch(`/api/zones/${encodeURIComponent(zoneId)}`, {
      method: 'DELETE'
    });
    if (res.ok) {
      openToast(`Zona berhasil dihapus!`);
      fetchZonesList();
    }
  } catch (e) {
    openToast('Gagal menghapus zona.');
  }
}

async function clearAllZones() {
  if (!confirm('Apakah kamu yakin ingin menghapus semua zona deteksi?')) return;
  try {
    const res = await fetch('/api/zones', { method: 'DELETE' });
    if (res.ok) {
      openToast('Semua zona telah dihapus.');
      fetchZonesList();
    }
  } catch (e) {
    openToast('Gagal menghapus semua zona.');
  }
}

async function resetDefaultZones() {
  try {
    const res = await fetch('/api/zones/reset', { method: 'POST' });
    if (res.ok) {
      openToast('Zona dikembalikan ke pengaturan default (A, B, C).');
      fetchZonesList();
    }
  } catch (e) {
    openToast('Gagal me-reset zona.');
  }
}

// CHART.JS INITIALIZATION & REALTIME UPDATE
function initTrendChart() {
  const ctxEl = document.getElementById('crowdTrendChart');
  if (!ctxEl) return;

  const now = new Date();
  for (let i = 8; i >= 0; i--) {
    const t = new Date(now.getTime() - i * 15000);
    historyLabels.push(t.toTimeString().split(' ')[0]);
    historyTotal.push(0);
  }

  trendChart = new Chart(ctxEl, {
    type: 'line',
    data: {
      labels: historyLabels,
      datasets: [
        {
          label: 'Total Dalam Zona',
          data: historyTotal,
          borderColor: '#ffffff',
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.35,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(10, 20, 35, 0.95)',
          titleColor: '#00e5ff',
          bodyColor: '#fff',
          borderColor: '#1e3a5f',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30, 58, 95, 0.25)' },
          ticks: { color: '#566b88', font: { size: 9, family: 'JetBrains Mono' }, maxTicksLimit: 6 }
        },
        y: {
          grid: { color: 'rgba(30, 58, 95, 0.25)' },
          ticks: { color: '#566b88', font: { size: 9, family: 'JetBrains Mono' } },
          beginAtZero: true
        }
      }
    }
  });
}

let lastChartUpdate = 0;
function appendChartData(total, zones) {
  const now = Date.now();
  if (now - lastChartUpdate < 2000 || !trendChart) return;
  lastChartUpdate = now;

  const timeStr = new Date().toTimeString().split(' ')[0];
  if (historyLabels.length > 15) {
    historyLabels.shift();
    historyTotal.shift();
  }

  historyLabels.push(timeStr);
  historyTotal.push(total);

  // Sync dynamic zone datasets in chart
  if (zones && zones.length > 0) {
    zones.forEach(z => {
      if (!dynamicZoneHistories[z.id]) {
        dynamicZoneHistories[z.id] = new Array(historyLabels.length - 1).fill(0);
        // Add new dataset to chart
        trendChart.data.datasets.push({
          id: z.id,
          label: z.name,
          data: dynamicZoneHistories[z.id],
          borderColor: z.hex_color || '#00b4d8',
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.35,
          fill: false
        });
      }

      if (dynamicZoneHistories[z.id].length > 15) {
        dynamicZoneHistories[z.id].shift();
      }
      dynamicZoneHistories[z.id].push(z.count);
    });

    // Remove obsolete datasets
    const currentZoneIds = new Set(zones.map(z => z.id));
    trendChart.data.datasets = trendChart.data.datasets.filter(ds => {
      if (!ds.id) return true; // Keep Total
      return currentZoneIds.has(ds.id);
    });
  }

  trendChart.update('none');
}

// ALERT SYSTEM
let lastAlertTime = 0;
function triggerHighDensityAlert(zones, total) {
  const now = Date.now();
  if (now - lastAlertTime < 25000) return;
  lastAlertTime = now;

  const timeStr = new Date().toTimeString().split(' ')[0];
  const alertsList = document.getElementById('alerts-list');
  if (!alertsList) return;

  const highestZone = zones.reduce((prev, curr) => (curr.count > (prev?.count || 0)) ? curr : prev, null);
  const zoneName = highestZone ? highestZone.name : "Zona";

  const alertItem = document.createElement('div');
  alertItem.className = 'alert-item alert-critical';
  alertItem.innerHTML = `
    <div class="alert-icon"><i data-lucide="alert-octagon"></i></div>
    <div class="alert-body">
      <div class="alert-title">Kepadatan Tinggi di ${zoneName}!</div>
      <div class="alert-meta">${timeStr} • ${highestZone ? highestZone.count : total} orang terdeteksi</div>
    </div>
  `;
  alertsList.insertBefore(alertItem, alertsList.firstChild);
  if (window.lucide) lucide.createIcons();
  openToast(`PERINGATAN: Kepadatan tinggi terdeteksi di ${zoneName}!`);
}

function clearAlerts() {
  const alertsList = document.getElementById('alerts-list');
  if (alertsList) {
    alertsList.innerHTML = `
      <div class="alert-item alert-success">
        <div class="alert-icon"><i data-lucide="check-circle-2"></i></div>
        <div class="alert-body">
          <div class="alert-title">Sistem Normal</div>
          <div class="alert-meta">Log dibersihkan oleh operator</div>
        </div>
      </div>
    `;
    if (window.lucide) lucide.createIcons();
  }
}

// TOGGLE CONTROLS
async function toggleBboxes() {
  showBoxes = !showBoxes;
  const btn = document.getElementById('btn-toggle-boxes');
  if (btn) {
    btn.innerHTML = `<i data-lucide="box"></i> Box: ${showBoxes ? 'ON' : 'OFF'}`;
    btn.style.color = showBoxes ? 'var(--primary-cyan)' : 'var(--text-muted)';
  }
  
  const form = new FormData();
  form.append('show_boxes', showBoxes);
  await fetch('/api/settings', { method: 'POST', body: form });
  if (window.lucide) lucide.createIcons();
  openToast(`Bounding Boxes: ${showBoxes ? 'Enabled' : 'Disabled'}`);
}

async function togglePoints() {
  showPoints = !showPoints;
  const btn = document.getElementById('btn-toggle-points');
  if (btn) {
    btn.innerHTML = `<i data-lucide="crosshair"></i> Dots: ${showPoints ? 'ON' : 'OFF'}`;
    btn.style.color = showPoints ? 'var(--primary-cyan)' : 'var(--text-muted)';
  }

  const form = new FormData();
  form.append('show_points', showPoints);
  await fetch('/api/settings', { method: 'POST', body: form });
  if (window.lucide) lucide.createIcons();
  openToast(`Head Dots (P2PNet): ${showPoints ? 'Enabled' : 'Disabled'}`);
}

async function toggleZones() {
  showZones = !showZones;
  const btn = document.getElementById('btn-toggle-zones');
  if (btn) {
    btn.innerHTML = `<i data-lucide="grid"></i> Zones: ${showZones ? 'ON' : 'OFF'}`;
    btn.style.color = showZones ? 'var(--primary-cyan)' : 'var(--text-muted)';
  }
  
  const form = new FormData();
  form.append('show_zones', showZones);
  await fetch('/api/settings', { method: 'POST', body: form });
  if (window.lucide) lucide.createIcons();
  openToast(`Zone Overlay: ${showZones ? 'Enabled' : 'Disabled'}`);
}

async function toggleHeatmap() {
  showHeatmap = !showHeatmap;
  const btn = document.getElementById('btn-toggle-heatmap');
  if (btn) {
    btn.innerHTML = `<i data-lucide="flame"></i> Heatmap: ${showHeatmap ? 'ON' : 'OFF'}`;
    if (showHeatmap) {
      btn.classList.add('btn-active-heatmap');
    } else {
      btn.classList.remove('btn-active-heatmap');
    }
  }

  try {
    await fetch('/api/crowd/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ show_heatmap: showHeatmap })
    });
  } catch (e) {
    console.error("toggleHeatmap error", e);
  }

  if (window.lucide) lucide.createIcons();
  openToast(`Density Heatmap Overlay: ${showHeatmap ? 'Enabled' : 'Disabled'}`);
}

async function setCrowdMode(mode) {
  currentCrowdMode = mode;
  document.querySelectorAll('.ce-mode-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`ce-mode-${mode}`);
  if (btn) btn.classList.add('active');

  let reqBody = { mode: mode, show_heatmap: showHeatmap };

  if (mode === 'density') {
    showHeatmap = true;
    const heatBtn = document.getElementById('btn-toggle-heatmap');
    if (heatBtn) {
      heatBtn.innerHTML = '<i data-lucide="flame"></i> Heatmap: ON';
      heatBtn.classList.add('btn-active-heatmap');
    }
    reqBody.show_heatmap = true;
  } else if (mode === 'p2pnet') {
    reqBody.engine = 'p2pnet';
    reqBody.show_points = true;
    showPoints = true;
    const dotBtn = document.getElementById('btn-toggle-points');
    if (dotBtn) {
      dotBtn.innerHTML = '<i data-lucide="crosshair"></i> Dots: ON';
      dotBtn.style.color = 'var(--primary-cyan)';
    }
  } else if (mode === 'detection') {
    reqBody.engine = 'yolo';
  }

  try {
    await fetch('/api/crowd/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody)
    });
  } catch (e) {
    console.error("setCrowdMode error", e);
  }

  if (window.lucide) lucide.createIcons();
  openToast(`Crowd Engine Mode: ${mode.toUpperCase()}`);
}

async function triggerCrowdEyeScan() {
  openToast("Memulai Deep Scan Crowd Eye...");
  try {
    const res = await fetch('/api/crowd/scan', { method: 'POST' });
    if (res.ok) {
      openToast("Deep scan selesai & telemetri diperbarui!");
    }
  } catch (e) {
    console.error("Deep scan error", e);
  }
}

// UPLOAD MODAL & HANDLER
function openUploadModal() {
  document.getElementById('upload-modal').classList.add('open');
  fetchVideosList();
}

function closeUploadModal() {
  document.getElementById('upload-modal').classList.remove('open');
}

function setupDropzone() {
  const dropzone = document.getElementById('dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleSelectedFile(files[0]);
    }
  });
}

function handleFileSelected(event) {
  const file = event.target.files[0];
  if (file) {
    handleSelectedFile(file);
  }
}

function handleSelectedFile(file) {
  selectedFile = file;
  document.getElementById('dropzone-text').textContent = `File dipilih: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
  document.getElementById('btn-submit-upload').disabled = false;
}

async function startUpload() {
  if (!selectedFile) return;

  const btn = document.getElementById('btn-submit-upload');
  const progressWrap = document.getElementById('upload-progress');
  const progressBar = document.getElementById('upload-bar');
  const progressLabel = document.getElementById('upload-label');

  btn.disabled = true;
  progressWrap.style.display = 'block';

  const formData = new FormData();
  formData.append('file', selectedFile);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload', true);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      progressBar.style.width = `${pct}%`;
      progressLabel.textContent = `Mengunggah & memproses video... ${pct}%`;
    }
  };

  xhr.onload = () => {
    if (xhr.status === 200) {
      const res = JSON.parse(xhr.responseText);
      openToast(`Video berhasil diunggah: ${res.filename}`);
      closeUploadModal();
      startStream();
      fetchVideosList();
    } else {
      openToast('Gagal mengunggah video. Pastikan format valid.');
    }
    btn.disabled = false;
    progressWrap.style.display = 'none';
  };

  xhr.onerror = () => {
    openToast('Terjadi kesalahan jaringan saat upload.');
    btn.disabled = false;
    progressWrap.style.display = 'none';
  };

  xhr.send(formData);
}

async function fetchVideosList() {
  try {
    const res = await fetch('/api/videos');
    if (!res.ok) return;
    const data = await res.json();
    const listEl = document.getElementById('sample-videos-list');
    if (!listEl) return;

    if (data.videos.length === 0) {
      listEl.innerHTML = '<div class="text-dim">Belum ada video lain. Silakan unggah rekaman.</div>';
      return;
    }

    listEl.innerHTML = data.videos.map(v => `
      <button class="sample-chip ${v.is_active ? 'active' : ''}" onclick="selectVideo('${v.name}')">
        <i data-lucide="play-circle"></i>
        <span>${v.name} (${v.size_mb} MB) ${v.is_active ? '<strong>[Sedang Aktif]</strong>' : ''}</span>
      </button>
    `).join('');

    if (window.lucide) lucide.createIcons();
  } catch (e) {}
}

async function selectVideo(name) {
  try {
    const res = await fetch(`/api/select-video/${encodeURIComponent(name)}`, { method: 'POST' });
    if (res.ok) {
      openToast(`Beralih ke video: ${name}`);
      closeUploadModal();
      setTimeout(startStream, 250);
      fetchVideosList();
    }
  } catch (e) {
    openToast('Gagal memuat video.');
  }
}

function loadSampleVideo() {
  selectVideo('demo_crowd.mp4');
}

// QUICK ACTIONS
function takeSnapshot() {
  window.open('/api/snapshot', '_blank');
  openToast('Snapshot berhasil diambil dan diunduh.');
}

function toggleRecording() {
  isRecording = !isRecording;
  openToast(isRecording ? 'Perekaman dimulai.' : 'Perekaman disimpan.');
}

function toggleEmergency() {
  isEmergency = !isEmergency;
  const container = document.querySelector('.command-container');
  if (isEmergency) {
    container.style.boxShadow = 'inset 0 0 80px rgba(255, 51, 75, 0.4)';
    openToast('MODE DARURAT DIAKTIFKAN!');
  } else {
    container.style.boxShadow = 'none';
    openToast('Mode darurat dinonaktifkan.');
  }
}

function exportData() {
  const data = {
    timestamp: new Date().toISOString(),
    total_people_in_zones: document.getElementById('stat-total-people').textContent,
    active_zones: activeZones
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `crowdvision_report_${Date.now()}.json`;
  a.click();
  openToast('Laporan Crowd Analytics berhasil diekspor.');
}

function toggleFullscreen() {
  const vp = document.getElementById('video-viewport');
  if (!document.fullscreenElement) {
    vp.requestFullscreen().catch(err => alert(`Error fullscreen: ${err.message}`));
  } else {
    document.exitFullscreen();
  }
}

// SETTINGS MODAL
function openSettingsModal() {
  document.getElementById('settings-modal').classList.add('open');
}

function closeSettingsModal() {
  document.getElementById('settings-modal').classList.remove('open');
}

function updateConf(val) {
  document.getElementById('conf-val').textContent = parseFloat(val).toFixed(2);
}

async function saveSettings() {
  const conf = parseFloat(document.getElementById('conf-slider').value);
  const threshold = parseInt(document.getElementById('setting-threshold').value, 10);
  const modelSelect = document.getElementById('setting-model-select');
  const selectedModel = modelSelect ? modelSelect.value : null;
  
  const form = new FormData();
  form.append('conf', conf);
  form.append('alert_threshold', threshold);
  if (selectedModel) {
    form.append('model_name', selectedModel);
    if (selectedModel.includes('p2p') || selectedModel.endsWith('.pth')) {
      form.append('engine', 'p2pnet');
    } else {
      form.append('engine', 'yolo');
    }
  }

  await fetch('/api/settings', { method: 'POST', body: form });
  openToast('Konfigurasi deteksi & model diperbarui.');
  closeSettingsModal();
}

// TOGGLE RIGHT SIDEBAR PANEL (MAP & TELEMETRY)
function toggleRightPanel() {
  const panel = document.querySelector('.right-panel');
  if (!panel) return;
  panel.classList.toggle('collapsed');
  const isCollapsed = panel.classList.contains('collapsed');
  const btn = document.getElementById('btn-toggle-right-panel');
  if (btn) {
    if (isCollapsed) {
      btn.style.opacity = '0.7';
    } else {
      btn.style.opacity = '1';
    }
  }
  openToast(isCollapsed ? 'Panel Peta disembunyikan' : 'Panel Peta ditampilkan');
  setTimeout(() => {
    window.dispatchEvent(new Event('resize'));
  }, 280);
}

// TOAST
let toastTimeout = null;
function openToast(msg) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
}
