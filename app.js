// app.js (updated: metadata overlay + file loader integration)
// Place next to interactive_network.html. This file is the previous version enhanced with
// automatic updates of the metadata overlay (source/filename, node count, edge count).

(function () {
  // ---------------------------
  // Configurable sizing params
  // ---------------------------
  const REF_N = 1000;
  const REF_SIZE = 6;
  const MIN_SIZE = 2;
  const MAX_SIZE = 24;
  const TARGET_SPAN = 1200;

  // ---- helper graph generators ----

  function generateSparseGraph(n, avgDeg) {
    const nodes = [];
    const edges = [];
    for (let i = 0; i < n; i++) nodes.push({ data: { id: 'n' + i } });

    const m = Math.max(0, Math.round(n * avgDeg / 2));
    const edgeSet = new Set();
    const randInt = (a, b) => Math.floor(Math.random() * (b - a + 1)) + a;

    while (edgeSet.size < m) {
      const u = randInt(0, n - 1);
      const v = randInt(0, n - 1);
      if (u === v) continue;
      const key = u < v ? `${u}|${v}` : `${v}|${u}`;
      if (edgeSet.has(key)) continue;
      edgeSet.add(key);
      edges.push({ data: { id: 'e' + edgeSet.size, source: 'n' + u, target: 'n' + v } });
    }
    return { nodes, edges };
  }

  function generateBarabasiAlbert(n, m) {
    if (m >= n) m = Math.max(1, n - 1);
    const nodes = [];
    const edges = [];
    for (let i = 0; i < n; i++) nodes.push({ data: { id: 'n' + i } });

    const degree = new Array(n).fill(0);
    const edgeSet = new Set();
    const addEdge = (u, v) => {
      if (u === v) return;
      const a = Math.min(u, v), b = Math.max(u, v);
      const key = `${a}|${b}`;
      if (edgeSet.has(key)) return;
      edgeSet.add(key);
      edges.push({ data: { id: 'e' + edgeSet.size, source: 'n' + a, target: 'n' + b } });
      degree[a]++; degree[b]++;
    };

    if (m === 1) {
      addEdge(0, 1);
      var next = 2;
    } else {
      for (let i = 0; i < m; i++) {
        for (let j = i + 1; j < m; j++) addEdge(i, j);
      }
      var next = m;
    }

    const randChoiceByDegree = () => {
      const cum = [];
      let s = 0;
      for (let i = 0; i < next; i++) {
        s += Math.max(1, degree[i]);
        cum.push(s);
      }
      const total = s;
      const pick = Math.random() * total;
      let lo = 0, hi = cum.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (pick < cum[mid]) hi = mid;
        else lo = mid + 1;
      }
      return lo;
    };

    while (next < n) {
      const targets = new Set();
      const limit = Math.min(m, next);
      let attempts = 0;
      while (targets.size < limit && attempts < limit * 20) {
        const v = randChoiceByDegree();
        if (v >= next) { attempts++; continue; }
        targets.add(v);
        attempts++;
      }
      if (targets.size < limit) {
        for (let cand = 0; cand < next && targets.size < limit; cand++) targets.add(cand);
      }
      for (const t of targets) addEdge(next, t);
      next++;
    }

    return { nodes, edges };
  }

  // ---- position helpers ----

  function makePresetPositions(nodes) {
    const pos = {};
    for (let i = 0; i < nodes.length; i++) {
      pos[nodes[i].data.id] = { x: (Math.random() - 0.5) * 2000, y: (Math.random() - 0.5) * 2000 };
    }
    return pos;
  }

  function makeGridPositions(nodes) {
    const pos = {};
    const n = nodes.length;
    const cols = Math.ceil(Math.sqrt(n));
    const gap = 25;
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      pos[nodes[i].data.id] = { x: c * gap, y: r * gap };
    }
    return pos;
  }

  function makeCirclePositions(nodes) {
    const pos = {};
    const n = nodes.length;
    const R = Math.max(300, n * 0.5);
    for (let i = 0; i < n; i++) {
      const a = 2 * Math.PI * i / n;
      pos[nodes[i].data.id] = { x: Math.cos(a) * R, y: Math.sin(a) * R };
    }
    return pos;
  }

  function normalizePositions(positions, targetSpan = TARGET_SPAN) {
    const ids = Object.keys(positions);
    if (ids.length === 0) return positions;
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (const id of ids) {
      const p = positions[id];
      if (p.x < minx) minx = p.x;
      if (p.x > maxx) maxx = p.x;
      if (p.y < miny) miny = p.y;
      if (p.y > maxy) maxy = p.y;
    }
    const spanX = maxx - minx;
    const spanY = maxy - miny;
    const span = Math.max(spanX, spanY, 1e-6);
    const scale = targetSpan / span;

    let cx = 0, cy = 0;
    for (const id of ids) { cx += positions[id].x; cy += positions[id].y; }
    cx /= ids.length; cy /= ids.length;

    const out = {};
    for (const id of ids) {
      const p = positions[id];
      out[id] = { x: (p.x - cx) * scale + cx, y: (p.y - cy) * scale + cy };
    }
    return out;
  }

  function computeNodeSize(n) {
    if (n <= 0) return REF_SIZE;
    const raw = REF_SIZE * Math.sqrt(REF_N / n);
    const clamped = Math.max(MIN_SIZE, Math.min(MAX_SIZE, raw));
    return Math.round(clamped);
  }

  // ---- Cytoscape init ----

  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],
    style: [
      {
        selector: 'node',
        style: {
          'background-color': '#2b8cbe',
          'width': REF_SIZE,
          'height': REF_SIZE,
          'label': 'data(label)',
          'font-size': 9,
          'color': '#222',
          'text-valign': 'top',
          'text-halign': 'center',
          'text-margin-y': -8,
          'text-opacity': 0.0
        }
      },
      {
        selector: 'edge',
        style: {
          'line-color': '#bbb',
          'width': 1,
          'curve-style': 'bezier'
        }
      },
      {
        selector: '.highlight',
        style: {
          'background-color': '#f03b20',
          'width': REF_SIZE * 1.6,
          'height': REF_SIZE * 1.6
        }
      }
    ],
    layout: { name: 'preset' },
    wheelSensitivity: 0.2,
    motionBlur: true,
    textureOnViewport: true
  });

  // ---- state & UI nodes ----

  let currentData = null;
  let currentPositions = null;
  let labelsVisible = false;
  let lastFileName = null; // store last loaded filename for metadata

  const nodeCountInput = document.getElementById('nodeCount');
  const avgDegInput = document.getElementById('avgDeg');
  const baNodeCountInput = document.getElementById('baNodeCount');
  const baMInput = document.getElementById('baM');

  const btnGenerate = document.getElementById('btnGenerate');
  const btnApplyLayout = document.getElementById('btnApplyLayout');
  const layoutSelect = document.getElementById('layoutSelect');
  const btnFit = document.getElementById('btnFit');
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  const btnToggleLabels = document.getElementById('btnToggleLabels');
  const btnReset = document.getElementById('btnReset');
  const searchInput = document.getElementById('searchId');
  const btnSearch = document.getElementById('btnSearch');

  const btnLoadFile = document.getElementById('btnLoadFile');
  const fileInput = document.getElementById('fileInput');

  const modeRandomBtn = document.getElementById('modeRandom');
  const modeBABtn = document.getElementById('modeBA');
  const randomControls = document.getElementById('randomControls');
  const baControls = document.getElementById('baControls');

  const metaSource = document.getElementById('metaSource');
  const metaNodes = document.getElementById('metaNodes');
  const metaEdges = document.getElementById('metaEdges');

  // ---- UI mode switching ----

  function setModeRandom() {
    modeRandomBtn.classList.add('active');
    modeBABtn.classList.remove('active');
    randomControls.style.display = '';
    baControls.style.display = 'none';
  }
  function setModeBA() {
    modeBABtn.classList.add('active');
    modeRandomBtn.classList.remove('active');
    randomControls.style.display = 'none';
    baControls.style.display = '';
  }
  modeRandomBtn.addEventListener('click', setModeRandom);
  modeBABtn.addEventListener('click', setModeBA);

  // ---- file parsing logic ----

  function parseSimpleEdgeFile(text) {
    const lines = text.split(/\r?\n/);
    let N = null;
    const edges = [];
    for (let raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      if (line.startsWith('#') || line.startsWith('c') || line.startsWith('%')) continue;
      const parts = line.split(/\s+/);
      if (parts.length === 0) continue;
      const head = parts[0].toLowerCase();
      if (head === 'p' && parts.length >= 3) {
        const maybeN = parseInt(parts[2], 10);
        if (!isNaN(maybeN) && maybeN > 0) N = maybeN;
      } else if (head === 'e' && parts.length >= 3) {
        const a = parseInt(parts[1], 10);
        const b = parseInt(parts[2], 10);
        if (!isNaN(a) && !isNaN(b)) edges.push([a, b]);
      } else {
        if (parts.length >= 2) {
          const a = parseInt(parts[0], 10);
          const b = parseInt(parts[1], 10);
          if (!isNaN(a) && !isNaN(b)) edges.push([a, b]);
        }
      }
    }
    return { nodesCount: N, edges };
  }

  function buildGraphFromParsed(parsed) {
    let N = parsed.nodesCount;
    let edges = parsed.edges;
    if (N == null) {
      let maxIndex = 0;
      for (const [a, b] of edges) {
        if (a > maxIndex) maxIndex = a;
        if (b > maxIndex) maxIndex = b;
      }
      N = Math.max(0, maxIndex);
    }
    const nodes = [];
    for (let i = 0; i < N; i++) nodes.push({ data: { id: 'n' + i } });

    const edgeSet = new Set();
    const edgesOut = [];
    let eid = 1;
    for (const [a1, b1] of edges) {
      const a = a1 - 1;
      const b = b1 - 1;
      if (a < 0 || b < 0 || a >= N || b >= N) continue;
      if (a === b) continue;
      const key = a < b ? `${a}|${b}` : `${b}|${a}`;
      if (edgeSet.has(key)) continue;
      edgeSet.add(key);
      edgesOut.push({ data: { id: 'e' + (eid++), source: 'n' + a, target: 'n' + b } });
    }
    return { nodes, edges: edgesOut };
  }

  // ---- styling helpers ----

  function applyNodeStylingForN(n) {
    const size = computeNodeSize(n);
    cy.style()
      .selector('node')
      .style('width', size)
      .style('height', size)
      .style('text-margin-y', -Math.round(size * 1.1))
      .style('font-size', Math.max(8, Math.round(size * 0.9)));
    cy.style()
      .selector('.highlight')
      .style('width', Math.round(size * 1.6))
      .style('height', Math.round(size * 1.6));
    cy.style().update();
  }

  // ---- metadata updater ----

  function updateMeta(sourceLabel, nNodes, nEdges, filename) {
    const label = filename ? `${filename}` : sourceLabel;
    metaSource.textContent = label;
    metaNodes.textContent = String(nNodes);
    metaEdges.textContent = String(nEdges);
  }

  function loadAndRender(data, presetType = 'preset', sourceLabel = 'Generated', filename = null) {
    cy.elements().remove();
    cy.style().selector('node').style('text-opacity', labelsVisible ? 1.0 : 0.0);
    cy.add(data);

    let rawPos = null;
    if (presetType === 'preset') rawPos = makePresetPositions(data.nodes);
    else if (presetType === 'grid') rawPos = makeGridPositions(data.nodes);
    else if (presetType === 'circle') rawPos = makeCirclePositions(data.nodes);
    else rawPos = makePresetPositions(data.nodes);

    const pos = normalizePositions(rawPos, TARGET_SPAN);

    for (const id in pos) {
      const n = cy.getElementById(id);
      if (n) n.position(pos[id]);
    }

    cy.nodes().forEach((ele) => {
      ele.data('label', ele.id());
    });

    const nNodes = data.nodes.length;
    const nEdges = data.edges.length;
    applyNodeStylingForN(nNodes);

    currentData = data;
    currentPositions = pos;
    cy.fit(20);

    lastFileName = filename || null;
    updateMeta(sourceLabel, nNodes, nEdges, filename);
  }

  // ---- UI wiring ----

  btnGenerate.addEventListener('click', () => {
    let data = null;
    let sourceLabel = 'Generated';
    if (modeBABtn.classList.contains('active')) {
      const n = Math.max(10, Math.min(5000, parseInt(baNodeCountInput.value) || 1000));
      let m = Math.max(1, Math.min(50, parseInt(baMInput.value) || 1));
      if (m >= n) m = Math.max(1, n - 1);
      data = generateBarabasiAlbert(n, m);
      sourceLabel = `Barabási–Albert (m=${m})`;
    } else {
      const n = Math.max(10, Math.min(5000, parseInt(nodeCountInput.value) || 1000));
      const avg = Math.max(0, Math.min(100, parseFloat(avgDegInput.value) || 4));
      data = generateSparseGraph(n, avg);
      sourceLabel = `Random (avgDeg=${avg})`;
    }
    loadAndRender(data, layoutSelect.value === 'preset' ? 'preset' : layoutSelect.value, sourceLabel, null);
  });

  // file load wiring
  btnLoadFile.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      try {
        const parsed = parseSimpleEdgeFile(text);
        const graph = buildGraphFromParsed(parsed);
        if (graph.nodes.length === 0) {
          alert('No nodes found in file.');
          return;
        }
        nodeCountInput.value = graph.nodes.length;
        baNodeCountInput.value = graph.nodes.length;
        loadAndRender(graph, 'preset', 'Loaded from file', f.name || null);
      } catch (err) {
        alert('Failed to parse file: ' + err.message);
      }
    };
    reader.onerror = () => alert('Failed to read file');
    reader.readAsText(f);
    fileInput.value = '';
  });

  btnApplyLayout.addEventListener('click', () => {
    const choice = layoutSelect.value;
    if (choice === 'preset' || choice === 'grid' || choice === 'circle') {
      loadAndRender(currentData || generateSparseGraph(1000, 4), choice, lastFileName ? 'Loaded from file' : 'Generated', lastFileName);
    } else if (choice === 'cose') {
      const layout = cy.layout({ name: 'cose', animate: true, animationDuration: 800, randomize: true, nodeRepulsion: 4000, idealEdgeLength: 50, numIter: 400 });
      layout.run();
      layout.on('layoutstop', () => {
        const positions = {};
        cy.nodes().forEach((n) => { positions[n.id()] = n.position(); });
        const norm = normalizePositions(positions, TARGET_SPAN);
        for (const id in norm) {
          cy.getElementById(id).position(norm[id]);
        }
        if (currentData) applyNodeStylingForN(currentData.nodes.length);
        cy.fit(20);
        // update meta after COSE as well (source stays same)
        if (currentData) updateMeta(lastFileName ? 'Loaded from file' : 'Generated', currentData.nodes.length, currentData.edges.length, lastFileName);
      });
    }
  });

  btnFit.addEventListener('click', () => cy.fit(20));
  btnZoomIn.addEventListener('click', () => cy.zoom({ level: cy.zoom() * 1.2, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }));
  btnZoomOut.addEventListener('click', () => cy.zoom({ level: cy.zoom() * 0.8, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }));

  btnToggleLabels.addEventListener('click', () => {
    labelsVisible = !labelsVisible;
    cy.style().selector('node').style('text-opacity', labelsVisible ? 1.0 : 0.0);
    cy.style().update();
  });

  btnReset.addEventListener('click', () => {
    if (currentData) loadAndRender(currentData, 'preset', lastFileName ? 'Loaded from file' : 'Generated', lastFileName);
  });

  btnSearch.addEventListener('click', () => {
    const q = searchInput.value.trim();
    if (!q) return;
    const el = cy.getElementById(q);
    if (el && el.length) {
      cy.elements().removeClass('highlight');
      el.addClass('highlight');
      cy.animate({ center: { eles: el }, duration: 400 });
    } else {
      alert('Node not found: ' + q);
    }
  });

  cy.on('tap', function (evt) {
    if (evt.target === cy) {
      cy.elements().removeClass('highlight');
    }
  });

  window.cy = cy;

  // initial sample generation
  loadAndRender(generateSparseGraph(1000, 4), 'preset', 'Random (avgDeg=4)', null);
})();
