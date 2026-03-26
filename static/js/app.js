/**
 * app.js — Fractal Visualizer frontend controller
 * Supports: mandelbrot, julia, burning_ship, newton, lyapunov, ifs,
 *           tricorn, multibrot, phoenix, burning_ship_julia, nova, collatz, buddhabrot
 */
(function () {
  'use strict';

  const API = '';

  // ── Default viewports per fractal type ──────────────────────────────
  const DEFAULTS = {
    mandelbrot:          { xmin:-2.5,  xmax:1.0,   ymin:-1.25, ymax:1.25  },
    julia:               { xmin:-2.0,  xmax:2.0,   ymin:-1.5,  ymax:1.5   },
    burning_ship:        { xmin:-2.5,  xmax:1.5,   ymin:-2.0,  ymax:0.5   },
    newton:              { xmin:-2.0,  xmax:2.0,   ymin:-2.0,  ymax:2.0   },
    lyapunov:            { xmin:2.0,   xmax:4.0,   ymin:2.0,   ymax:4.0   },
    tricorn:             { xmin:-2.5,  xmax:1.0,   ymin:-1.25, ymax:1.25  },
    multibrot:           { xmin:-2.0,  xmax:2.0,   ymin:-2.0,  ymax:2.0   },
    phoenix:             { xmin:-2.0,  xmax:2.0,   ymin:-2.0,  ymax:2.0   },
    burning_ship_julia:  { xmin:-2.0,  xmax:2.0,   ymin:-2.0,  ymax:2.0   },
    nova:                { xmin:-2.0,  xmax:2.0,   ymin:-2.0,  ymax:2.0   },
    collatz:             { xmin:-3.0,  xmax:3.0,   ymin:-3.0,  ymax:3.0   },
    buddhabrot:          { xmin:-2.5,  xmax:1.0,   ymin:-1.25, ymax:1.25  },
  };

  // Fractal descriptions shown in the info panel
  const FRACTAL_INFO = {
    mandelbrot:         'Classic Mandelbrot set. z → z² + c. The most famous fractal.',
    julia:              'Julia set with fixed c. Vary Re(c) and Im(c) to explore infinite family of shapes.',
    burning_ship:       'z → (|Re(z)| + i|Im(z)|)² + c. The "ships" appear near y ≈ −1.7.',
    newton:             'Newton\'s method on zⁿ − 1. Basins of attraction create colour regions. Degree 3–8.',
    lyapunov:           'Lyapunov exponent map: blue=stable (λ<0), red=chaotic (λ>0). Sequence AB,AABAB,…',
    tricorn:            'Mandelbar: z → conj(z)² + c. Conjugate iteration produces 3-fold symmetry.',
    multibrot:          'Generalized z → zⁿ + c with real exponent n. n=2 → Mandelbrot, n=3 → Multibrot3.',
    phoenix:            'z_{n+1} = z_n² + p + q·z_{n-1}. Memory term creates phoenix-wing structures.',
    burning_ship_julia: 'Julia variant of the Burning Ship. Fixed c, pixel → initial z. Try c = −1.755.',
    nova:               'Newton fractal + perturbation c. z → z − (zⁿ−1)/(n·zⁿ⁻¹) + c. Rich basin art.',
    collatz:            'Complex Collatz: f(z)=(1+4z−(1+2z)cos(πz))/4. Organic, spiralling structure.',
    buddhabrot:         'Density map of Mandelbrot escape trajectories. RGB channels = low/mid/high iter counts. Slow!',
  };

  // IFS presets [a, b, c, d, e, f, prob]
  const IFS_PRESETS = {
    barnsley:   [[0,0,0,0.16,0,0,0.01],[0.85,0.04,-0.04,0.85,0,1.6,0.85],[0.2,-0.26,0.23,0.22,0,1.6,0.07],[-0.15,0.28,0.26,0.24,0,0.44,0.07]],
    sierpinski: [[0.5,0,0,0.5,0,0,0.33],[0.5,0,0,0.5,0.5,0,0.33],[0.5,0,0,0.5,0.25,0.433,0.34]],
    dragon:     [[0.824,0.281,-0.212,0.864,-1.882,0.552,0.787803],[-0.177,0.543,-0.471,-0.22,-0.657,0.647,0.212197]],
    levy:       [[0.5,-0.5,0.5,0.5,0,0,0.5],[0.5,0.5,-0.5,0.5,0.5,0.5,0.5]],
    tree:       [[0,0,0,0.5,0,0,0.05],[0.42,-0.42,0.42,0.42,0,0.2,0.4],[-0.42,0.42,0.42,0.42,0,0.2,0.4],[0.1,0,0,0.1,0,0.2,0.15]],
  };

  // ── State ────────────────────────────────────────────────────────────
  let state = {
    fractalType: 'mandelbrot',
    xmin: -2.5, xmax: 1.0, ymin: -1.25, ymax: 1.25,
    maxIter: 256, width: 800, height: 600, threads: 4,
    palette: 'ultra', colorMode: 0, gamma: 1.0,
    cycleSpeed: 1.0, bailout: 2.0, invert: false,
    // Julia
    juliaCr: -0.7, juliaCi: 0.27015,
    // Newton / Nova
    newtonDeg: 3,
    // Lyapunov
    lyapunovSeq: 'AB', lyapunovWarmup: 100,
    // Multibrot
    multibrotExp: 3.0,
    // Phoenix
    phoenixCr: 0.5667, phoenixCi: -0.5,
    // Burning Ship Julia
    bsjCr: -1.755, bsjCi: 0.0,
    // Nova
    novaDeg: 3,
    // Buddhabrot
    brotIterR: 100, brotIterG: 1000, brotIterB: 5000, brotSamples: 2000000,
  };

  let zoomStack = [];
  let bookmarks = [];
  try { bookmarks = JSON.parse(localStorage.getItem('fv_bookmarks') || '[]'); } catch (_) {}
  let currentImageURL = null;

  // Animation
  let animRunning = false, animAngle = 0, animRAF = null;
  let animFrameCount = 0, animFpsAcc = 0;

  // Drag-to-zoom
  let dragStart = null, dragRect = null;

  // ── DOM refs ─────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);
  const canvas2d   = $('fractal-canvas');
  const ctx2d      = canvas2d.getContext('2d');
  const animCanvas = $('anim-canvas');
  const animCtx    = animCanvas.getContext('2d');
  const ifsCanvas  = $('ifs-canvas');
  const ifsCtx     = ifsCanvas.getContext('2d');
  const zoomBox    = $('zoom-box');
  const crossH     = $('crosshair-h');
  const crossV     = $('crosshair-v');
  const coordReadout = $('coord-readout');
  const progressBar  = $('progress-bar');

  // ════════════════════════════════════════════════════════════════════
  //  INIT
  // ════════════════════════════════════════════════════════════════════
  function init() {
    fetchInfo();
    bindControls();
    bindCanvasInteraction();
    renderBookmarks();
    loadStateFromURL();
    FractalSurface.init('three-container');
  }

  async function fetchInfo() {
    try {
      const r = await fetch(API + '/api/info');
      const d = await r.json();
      $('engine-badge').textContent = d.native_engine ? `C++ (${d.hardware_threads}T)` : 'Python';
      $('engine-badge').style.color = d.native_engine ? 'var(--accent)' : 'var(--accent2)';
      $('st-engine').textContent = d.native_engine ? `C++/pybind11 (${d.hardware_threads}T)` : 'Python fallback';
      $('p-threads').max = d.hardware_threads || 16;
    } catch (_) {
      $('engine-badge').textContent = 'Offline';
    }
  }

  // ════════════════════════════════════════════════════════════════════
  //  BIND CONTROLS
  // ════════════════════════════════════════════════════════════════════
  function bindControls() {
    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn =>
      btn.addEventListener('click', () => switchTab(btn.dataset.tab)));

    // Fractal type buttons
    document.querySelectorAll('.ftype-btn').forEach(btn =>
      btn.addEventListener('click', () => selectFractalType(btn.dataset.type)));

    // ── Sliders ──
    bindSlider('p-maxiter',     'lbl-maxiter',     v => state.maxIter = +v);
    bindSlider('p-threads',     'lbl-threads',     v => state.threads = +v);
    bindSlider('p-gamma',       'lbl-gamma',       v => state.gamma = +v, 2);
    bindSlider('p-cycle',       'lbl-cycle',       v => state.cycleSpeed = +v, 1);
    bindSlider('p-bailout',     'lbl-bailout',     v => state.bailout = +v, 1);
    // Julia
    bindSlider('p-julia-cr',    'lbl-julia-cr',    v => state.juliaCr = +v, 3);
    bindSlider('p-julia-ci',    'lbl-julia-ci',    v => state.juliaCi = +v, 3);
    // Newton
    bindSlider('p-newton-deg',  'lbl-newton-deg',  v => state.newtonDeg = +v);
    bindSlider('p-newton-tol',  'lbl-newton-tol',  v => {
      $('lbl-newton-tol').textContent = `1e${Math.round(+v)}`;
    });
    // Lyapunov
    bindSlider('p-lyapunov-wu', 'lbl-lyapunov-wu', v => state.lyapunovWarmup = +v);
    // Multibrot
    bindSlider('p-multibrot-exp','lbl-multibrot-exp', v => state.multibrotExp = +v, 1);
    // Phoenix
    bindSlider('p-phoenix-cr',  'lbl-phoenix-cr',  v => state.phoenixCr = +v, 3);
    bindSlider('p-phoenix-ci',  'lbl-phoenix-ci',  v => state.phoenixCi = +v, 3);
    // BSJ
    bindSlider('p-bsj-cr',      'lbl-bsj-cr',      v => state.bsjCr = +v, 3);
    bindSlider('p-bsj-ci',      'lbl-bsj-ci',      v => state.bsjCi = +v, 3);
    // Nova
    bindSlider('p-nova-deg',    'lbl-nova-deg',    v => state.novaDeg = +v);
    // Buddhabrot
    bindSlider('p-brot-ir',     'lbl-brot-ir',     v => state.brotIterR = +v);
    bindSlider('p-brot-ig',     'lbl-brot-ig',     v => state.brotIterG = +v);
    bindSlider('p-brot-ib',     'lbl-brot-ib',     v => state.brotIterB = +v);
    // 3D
    bindSlider('p-zscale',      'lbl-zscale',      () => {}, 1);
    // Animate
    bindSlider('anim-radius',   'lbl-anim-radius', () => {}, 2);
    bindSlider('anim-speed',    'lbl-anim-speed',  () => {}, 3);

    // ── Selects ──
    $('p-palette').addEventListener('change',   e => state.palette = e.target.value);
    $('p-colormode').addEventListener('change', e => state.colorMode = +e.target.value);
    $('p-width').addEventListener('change',     e => state.width = +e.target.value);
    $('p-height').addEventListener('change',    e => state.height = +e.target.value);
    $('p-brot-samples').addEventListener('change', e => state.brotSamples = +e.target.value);

    // ── Checkbox ──
    $('p-invert').addEventListener('change', e => state.invert = e.target.checked);

    // ── Viewport number inputs ──
    ['p-xmin','p-xmax','p-ymin','p-ymax'].forEach(id =>
      $(id).addEventListener('change', e => { state[id.slice(2)] = +e.target.value; }));

    // ── Text inputs ──
    $('p-lyapunov-seq').addEventListener('input', e => {
      state.lyapunovSeq = e.target.value.toUpperCase().replace(/[^AB]/g,'') || 'AB';
      e.target.value = state.lyapunovSeq;
    });

    // ── Action buttons ──
    $('btn-render').addEventListener('click',      () => render2D());
    $('btn-reset-view').addEventListener('click',  resetView);
    $('btn-download').addEventListener('click',    downloadPNG);
    $('btn-fullscreen').addEventListener('click',  toggleFullscreen);
    $('btn-undo-zoom').addEventListener('click',   undoZoom);
    $('btn-bookmark').addEventListener('click',    bookmarkView);
    $('btn-copy-url').addEventListener('click',    copyURL);
    $('btn-render-3d').addEventListener('click',   render3D);
    $('btn-anim-toggle').addEventListener('click', toggleAnimation);
    $('btn-julia-grid').addEventListener('click',  renderJuliaGrid);
    $('btn-ifs').addEventListener('click',         renderIFS);
  }

  function bindSlider(id, lblId, cb, decimals = 0) {
    const el = $(id), lbl = $(lblId);
    if (!el || !lbl) return;
    el.addEventListener('input', () => {
      const formatted = (+el.value).toFixed(decimals);
      lbl.textContent = formatted;
      cb(el.value);
    });
  }

  // ════════════════════════════════════════════════════════════════════
  //  TAB SWITCHING
  // ════════════════════════════════════════════════════════════════════
  function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === `tab-${tab}`));
    if (tab === '3d') FractalSurface.resize();
  }

  // ════════════════════════════════════════════════════════════════════
  //  FRACTAL TYPE SELECTION
  // ════════════════════════════════════════════════════════════════════
  // Map fractal type → which param panel IDs to show
  const PARAM_PANELS = {
    julia:              ['julia-params'],
    newton:             ['newton-params'],
    nova:               ['nova-params'],
    lyapunov:           ['lyapunov-params'],
    multibrot:          ['multibrot-params'],
    phoenix:            ['phoenix-params'],
    burning_ship_julia: ['bsj-params'],
    buddhabrot:         ['buddhabrot-params'],
  };

  function selectFractalType(type) {
    state.fractalType = type;
    document.querySelectorAll('.ftype-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.type === type));

    // Hide all conditional panels, then show ones for this type
    document.querySelectorAll('.ftype-params').forEach(p => p.style.display = 'none');
    (PARAM_PANELS[type] || []).forEach(id => {
      const el = $(id); if (el) el.style.display = 'block';
    });

    // Reset viewport to sensible default
    const def = DEFAULTS[type] || DEFAULTS.mandelbrot;
    Object.assign(state, def);
    updateViewportInputs();

    // Update info panel
    $('fractal-info-box').textContent = FRACTAL_INFO[type] || '';

    // Collatz uses a higher bailout by default
    if (type === 'collatz' && state.bailout < 16) {
      state.bailout = 32;
      $('p-bailout').value = 32;
      $('lbl-bailout').textContent = '32.0';
    }
    // Most others work at bailout 2
    if (type !== 'collatz' && state.bailout === 32) {
      state.bailout = 2;
      $('p-bailout').value = 2;
      $('lbl-bailout').textContent = '2.0';
    }
  }

  function updateViewportInputs() {
    $('p-xmin').value = state.xmin;
    $('p-xmax').value = state.xmax;
    $('p-ymin').value = state.ymin;
    $('p-ymax').value = state.ymax;
    const zoom = 3.5 / (state.xmax - state.xmin);
    $('st-zoom').textContent   = zoom.toFixed(1) + '×';
    $('st-center').textContent = `${((state.xmin+state.xmax)/2).toFixed(4)}, ${((state.ymin+state.ymax)/2).toFixed(4)}`;
  }

  // ════════════════════════════════════════════════════════════════════
  //  CANVAS MOUSE INTERACTION
  // ════════════════════════════════════════════════════════════════════
  function bindCanvasInteraction() {
    const wrapper = $('canvas-wrapper');

    wrapper.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      dragStart = { x: e.offsetX, y: e.offsetY };
      dragRect  = null;
      crossH.style.display = crossV.style.display = 'block';
    });

    wrapper.addEventListener('mousemove', e => {
      const px = e.offsetX, py = e.offsetY;
      const w = canvas2d.width, h = canvas2d.height;
      const cx = state.xmin + (px / w) * (state.xmax - state.xmin);
      const cy = state.ymax - (py / h) * (state.ymax - state.ymin);
      coordReadout.textContent = `${cx.toFixed(6)} + ${cy.toFixed(6)}i`;
      $('coord-detail').innerHTML =
        `Re: <span style="color:var(--accent)">${cx.toFixed(8)}</span><br>` +
        `Im: <span style="color:var(--accent2)">${cy.toFixed(8)}i</span>`;
      crossH.style.top  = py + 'px';
      crossV.style.left = px + 'px';

      if (dragStart) {
        const x0 = Math.min(dragStart.x, px), y0 = Math.min(dragStart.y, py);
        const bw = Math.abs(px - dragStart.x), bh = Math.abs(py - dragStart.y);
        if (bw > 5 || bh > 5) {
          zoomBox.style.cssText = `display:block;left:${x0}px;top:${y0}px;width:${bw}px;height:${bh}px;`;
          dragRect = { x0, y0, bw, bh };
        }
      }
    });

    wrapper.addEventListener('mouseup', e => {
      if (!dragStart) return;
      crossH.style.display = crossV.style.display = 'none';
      zoomBox.style.display = 'none';
      if (dragRect && dragRect.bw > 10 && dragRect.bh > 10) {
        zoomToRect(dragRect);
      } else {
        const w = canvas2d.width, h = canvas2d.height;
        const cx = state.xmin + (e.offsetX / w) * (state.xmax - state.xmin);
        const cy = state.ymax - (e.offsetY / h) * (state.ymax - state.ymin);
        zoomCentre(cx, cy, 0.4);
      }
      dragStart = dragRect = null;
    });

    wrapper.addEventListener('mouseleave', () => {
      crossH.style.display = crossV.style.display = 'none';
      if (dragStart) { zoomBox.style.display = 'none'; dragStart = dragRect = null; }
    });

    wrapper.addEventListener('wheel', e => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.3 : 0.77;
      const w = canvas2d.width, h = canvas2d.height;
      const cx = state.xmin + (e.offsetX / w) * (state.xmax - state.xmin);
      const cy = state.ymax - (e.offsetY / h) * (state.ymax - state.ymin);
      const hw = (state.xmax - state.xmin) * factor / 2;
      const hh = (state.ymax - state.ymin) * factor / 2;
      pushZoomStack();
      state.xmin = cx-hw; state.xmax = cx+hw; state.ymin = cy-hh; state.ymax = cy+hh;
      updateViewportInputs(); render2D();
    }, { passive: false });
  }

  function zoomToRect({ x0, y0, bw, bh }) {
    pushZoomStack();
    const w = canvas2d.width, h = canvas2d.height;
    const xr = state.xmax - state.xmin, yr = state.ymax - state.ymin;
    state.xmin = state.xmin + (x0/w)*xr;
    state.xmax = state.xmin + (bw/w)*xr + (x0/w)*xr - (x0/w)*xr; // recalc cleanly
    // simpler:
    const nxmin = DEFAULTS.mandelbrot.xmin; // placeholder — use direct calc below
    const nx0 = state.xmin + (x0/w)*xr;
    const nx1 = state.xmin + ((x0+bw)/w)*xr;
    const ny1 = state.ymax - (y0/h)*yr;
    const ny0 = state.ymax - ((y0+bh)/h)*yr;
    // undo placeholder and assign properly
    state.xmin = nx0; state.xmax = nx1; state.ymin = ny0; state.ymax = ny1;
    updateViewportInputs(); render2D();
  }

  function zoomCentre(cx, cy, ratio) {
    pushZoomStack();
    const hw = (state.xmax-state.xmin)*ratio, hh = (state.ymax-state.ymin)*ratio;
    state.xmin = cx-hw; state.xmax = cx+hw; state.ymin = cy-hh; state.ymax = cy+hh;
    updateViewportInputs(); render2D();
  }

  function pushZoomStack() {
    zoomStack.push({ xmin:state.xmin, xmax:state.xmax, ymin:state.ymin, ymax:state.ymax });
    if (zoomStack.length > 50) zoomStack.shift();
    renderZoomHistory();
  }

  function undoZoom() {
    if (!zoomStack.length) return;
    Object.assign(state, zoomStack.pop());
    updateViewportInputs(); render2D(); renderZoomHistory();
  }

  function resetView() {
    zoomStack = [];
    Object.assign(state, DEFAULTS[state.fractalType] || DEFAULTS.mandelbrot);
    updateViewportInputs(); render2D(); renderZoomHistory();
  }

  function renderZoomHistory() {
    const el = $('zoom-history'); el.innerHTML = '';
    [...zoomStack].reverse().slice(0, 8).forEach((z, i) => {
      const div = document.createElement('div');
      div.className = 'zoom-entry';
      div.textContent = `[${i+1}] x:[${z.xmin.toFixed(2)},${z.xmax.toFixed(2)}]`;
      div.addEventListener('click', () => {
        zoomStack = zoomStack.slice(0, zoomStack.length - 1 - i);
        Object.assign(state, z); updateViewportInputs(); render2D(); renderZoomHistory();
      });
      el.appendChild(div);
    });
  }

  // ════════════════════════════════════════════════════════════════════
  //  PARAMS BUILDER  — collects all current state into API payload
  // ════════════════════════════════════════════════════════════════════
  function buildParams() {
    return {
      fractal_type:    state.fractalType,
      xmin: state.xmin, xmax: state.xmax,
      ymin: state.ymin, ymax: state.ymax,
      width: state.width, height: state.height,
      max_iter:        state.maxIter,
      num_threads:     state.threads,
      palette:         state.palette,
      color_mode:      state.colorMode,
      gamma:           state.gamma,
      cycle_speed:     state.cycleSpeed,
      bailout:         state.bailout,
      invert:          state.invert,
      stripe_density:  5.0,
      // Julia
      julia_cr: state.juliaCr, julia_ci: state.juliaCi,
      // Newton / Nova
      newton_degree: state.newtonDeg,
      newton_tol: Math.pow(10, +($('p-newton-tol') ? $('p-newton-tol').value : -6)),
      nova_degree: state.novaDeg,
      nova_tol: 1e-6,
      // Lyapunov
      lyapunov_seq: state.lyapunovSeq, lyapunov_warmup: state.lyapunovWarmup,
      // Multibrot
      multibrot_exp: state.multibrotExp,
      // Phoenix
      phoenix_cr: state.phoenixCr, phoenix_ci: state.phoenixCi,
      // Burning Ship Julia
      bsj_cr: state.bsjCr, bsj_ci: state.bsjCi,
      // Buddhabrot
      brot_iter_r: state.brotIterR, brot_iter_g: state.brotIterG,
      brot_iter_b: state.brotIterB, brot_samples: state.brotSamples,
    };
  }

  // ════════════════════════════════════════════════════════════════════
  //  2D RENDER
  // ════════════════════════════════════════════════════════════════════
  async function render2D() {
    $('overlay-hint').style.display = 'none';
    const isBuddha = state.fractalType === 'buddhabrot';
    showLoading(isBuddha ? 'Buddhabrot — this may take a minute…' : 'Computing fractal…');
    setProgress(10);

    try {
      const r = await fetch(API + '/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildParams()),
      });
      setProgress(80);
      const d = await r.json();
      if (!d.ok) throw new Error(d.error || 'Unknown');

      const img = new Image();
      img.onload = () => {
        canvas2d.width = d.width; canvas2d.height = d.height;
        ctx2d.drawImage(img, 0, 0);
        setProgress(100);
        setTimeout(() => setProgress(0), 600);
        currentImageURL = d.image;
        $('render-time').textContent = `${d.render_ms} ms`;
        $('st-rtime').textContent = `${d.render_ms} ms`;
        $('st-res').textContent   = `${d.width} × ${d.height}`;
        updateViewportInputs();
        saveStateToURL();
      };
      img.src = d.image;
    } catch (err) {
      toast('Render error: ' + err.message, true);
    } finally {
      hideLoading();
    }
  }

  // ════════════════════════════════════════════════════════════════════
  //  3D RENDER
  // ════════════════════════════════════════════════════════════════════
  async function render3D() {
    showLoading('Building 3D surface…');
    const params = buildParams();
    params.z_scale   = +$('p-zscale').value;
    params.smooth_3d = $('p-smooth3d').checked;
    params.width = 200; params.height = 200;
    try {
      const r = await fetch(API + '/api/heightmap', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const d = await r.json();
      if (!d.ok) throw new Error(d.error || 'Unknown');
      FractalSurface.buildSurface(d.data, d.rows, d.cols, $('p-wireframe').checked);
      toast('3D surface rendered!');
    } catch (err) {
      toast('3D error: ' + err.message, true);
    } finally {
      hideLoading();
    }
  }

  // ════════════════════════════════════════════════════════════════════
  //  ANIMATION (Julia sweep)
  // ════════════════════════════════════════════════════════════════════
  function toggleAnimation() {
    if (animRunning) {
      animRunning = false;
      cancelAnimationFrame(animRAF);
      $('btn-anim-toggle').textContent = '▶ START';
      $('btn-anim-toggle').style.background = 'var(--accent)';
    } else {
      animRunning = true;
      $('btn-anim-toggle').textContent = '⏹ STOP';
      $('btn-anim-toggle').style.background = 'var(--accent2)';
      animStep();
    }
  }

  async function animStep() {
    if (!animRunning) return;
    const t0 = performance.now();
    const radius = +$('anim-radius').value;
    const speed  = +$('anim-speed').value;
    const aw     = +$('anim-width').value;
    const params = buildParams();
    params.fractal_type = 'julia';
    params.width = aw; params.height = Math.round(aw * 0.75);
    params.angle = animAngle; params.radius = radius;

    try {
      const r = await fetch(API + '/api/animation_frame', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const d = await r.json();
      if (d.ok) {
        animCanvas.width = d.width || aw;
        animCanvas.height = d.height || Math.round(aw * 0.75);
        const img = new Image();
        img.onload = () => animCtx.drawImage(img, 0, 0);
        img.src = d.image;
        const elapsed = performance.now() - t0;
        animFpsAcc += elapsed; animFrameCount++;
        if (animFrameCount >= 10) {
          $('anim-fps').textContent = (1000 / (animFpsAcc / animFrameCount)).toFixed(1) + ' fps';
          animFpsAcc = animFrameCount = 0;
        }
      }
    } catch (_) {}

    animAngle += speed;
    if (animRunning) animRAF = requestAnimationFrame(animStep);
  }

  // ════════════════════════════════════════════════════════════════════
  //  JULIA GRID
  // ════════════════════════════════════════════════════════════════════
  async function renderJuliaGrid() {
    showLoading('Generating Julia grid…');
    const params = buildParams();
    params.max_iter = Math.min(state.maxIter, 192);
    try {
      const r = await fetch(API + '/api/julia_preview_grid', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const d = await r.json();
      if (!d.ok) throw new Error(d.error || 'Unknown');

      const container = $('julia-grid-container');
      container.innerHTML = '';
      d.images.forEach(({ cr, ci, image }) => {
        const div = document.createElement('div');
        div.className = 'julia-tile';
        const img = document.createElement('img');
        img.src = image; img.loading = 'lazy';
        const lbl = document.createElement('div');
        lbl.className = 'julia-tile-label';
        lbl.textContent = `c = ${cr.toFixed(3)} + ${ci.toFixed(3)}i`;
        div.appendChild(img); div.appendChild(lbl);
        div.addEventListener('click', () => {
          selectFractalType('julia');
          state.juliaCr = cr; state.juliaCi = ci;
          $('p-julia-cr').value = cr; $('lbl-julia-cr').textContent = cr.toFixed(3);
          $('p-julia-ci').value = ci; $('lbl-julia-ci').textContent = ci.toFixed(3);
          switchTab('2d'); render2D();
        });
        container.appendChild(div);
      });
      toast('Julia grid rendered!');
    } catch (err) {
      toast('Grid error: ' + err.message, true);
    } finally {
      hideLoading();
    }
  }

  // ════════════════════════════════════════════════════════════════════
  //  IFS
  // ════════════════════════════════════════════════════════════════════
  async function renderIFS() {
    showLoading('Rendering IFS fractal…');
    const preset = $('ifs-preset').value;
    const params = {
      fractal_type:   'ifs', width: 800, height: 800,
      ifs_preset:     preset,
      ifs_points:     +$('ifs-points').value,
      ifs_transforms: IFS_PRESETS[preset] || IFS_PRESETS.barnsley,
      palette:        state.palette,
    };
    try {
      const r = await fetch(API + '/api/render', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const d = await r.json();
      if (!d.ok) throw new Error(d.error || 'Unknown');
      const img = new Image();
      img.onload = () => {
        ifsCanvas.width = d.width; ifsCanvas.height = d.height;
        ifsCtx.drawImage(img, 0, 0);
        toast(`IFS rendered in ${d.render_ms} ms`);
      };
      img.src = d.image;
    } catch (err) {
      toast('IFS error: ' + err.message, true);
    } finally {
      hideLoading();
    }
  }

  // ════════════════════════════════════════════════════════════════════
  //  BOOKMARKS
  // ════════════════════════════════════════════════════════════════════
  function bookmarkView() {
    const bm = { id: Date.now(),
                 label: `${state.fractalType} @ ${new Date().toLocaleTimeString()}`,
                 snapshot: { ...state } };
    bookmarks.unshift(bm);
    if (bookmarks.length > 20) bookmarks.pop();
    try { localStorage.setItem('fv_bookmarks', JSON.stringify(bookmarks)); } catch (_) {}
    renderBookmarks(); toast('Bookmark saved!');
  }

  function renderBookmarks() {
    const el = $('bookmark-list'); el.innerHTML = '';
    bookmarks.forEach(bm => {
      const div = document.createElement('div');
      div.className = 'bookmark-item';
      const span = document.createElement('span');
      span.textContent = bm.label; span.style.fontSize = '10px';
      const del = document.createElement('button');
      del.textContent = '×';
      del.addEventListener('click', e => { e.stopPropagation(); removeBookmark(bm.id); });
      div.appendChild(span); div.appendChild(del);
      div.addEventListener('click', () => loadBookmark(bm));
      el.appendChild(div);
    });
  }

  function removeBookmark(id) {
    bookmarks = bookmarks.filter(b => b.id !== id);
    try { localStorage.setItem('fv_bookmarks', JSON.stringify(bookmarks)); } catch (_) {}
    renderBookmarks();
  }

  function loadBookmark(bm) {
    Object.assign(state, bm.snapshot);
    selectFractalType(state.fractalType);
    updateViewportInputs(); syncAllInputs(); render2D();
  }

  // ════════════════════════════════════════════════════════════════════
  //  URL STATE
  // ════════════════════════════════════════════════════════════════════
  function saveStateToURL() {
    const p = new URLSearchParams({
      t:  state.fractalType,
      xn: state.xmin.toFixed(8), xx: state.xmax.toFixed(8),
      yn: state.ymin.toFixed(8), yx: state.ymax.toFixed(8),
      mi: state.maxIter, pal: state.palette, cm: state.colorMode,
      ga: state.gamma,   cr: state.juliaCr,  ci: state.juliaCi,
    });
    history.replaceState(null, '', '?' + p.toString());
  }

  function loadStateFromURL() {
    const p = new URLSearchParams(location.search);
    if (p.has('t')) state.fractalType = p.get('t');
    if (p.has('xn')) state.xmin = +p.get('xn');
    if (p.has('xx')) state.xmax = +p.get('xx');
    if (p.has('yn')) state.ymin = +p.get('yn');
    if (p.has('yx')) state.ymax = +p.get('yx');
    if (p.has('mi')) state.maxIter = +p.get('mi');
    if (p.has('pal')) state.palette = p.get('pal');
    if (p.has('cm')) state.colorMode = +p.get('cm');
    if (p.has('ga')) state.gamma = +p.get('ga');
    if (p.has('cr')) state.juliaCr = +p.get('cr');
    if (p.has('ci')) state.juliaCi = +p.get('ci');
    if (p.has('t')) selectFractalType(state.fractalType);
    updateViewportInputs(); syncAllInputs();
    if (p.has('t')) render2D();
  }

  function copyURL() {
    saveStateToURL();
    navigator.clipboard.writeText(location.href).then(() => toast('URL copied!'));
    const d = $('share-url-display');
    d.style.display = 'block'; d.textContent = location.href;
  }

  function syncAllInputs() {
    $('p-maxiter').value = state.maxIter; $('lbl-maxiter').textContent = state.maxIter;
    $('p-gamma').value   = state.gamma;   $('lbl-gamma').textContent   = state.gamma.toFixed(2);
    $('p-cycle').value   = state.cycleSpeed; $('lbl-cycle').textContent = state.cycleSpeed.toFixed(1);
    $('p-bailout').value = state.bailout; $('lbl-bailout').textContent = state.bailout.toFixed(1);
    $('p-palette').value    = state.palette;
    $('p-colormode').value  = state.colorMode;
    $('p-julia-cr').value   = state.juliaCr; $('lbl-julia-cr').textContent = state.juliaCr.toFixed(3);
    $('p-julia-ci').value   = state.juliaCi; $('lbl-julia-ci').textContent = state.juliaCi.toFixed(3);
    $('p-invert').checked   = state.invert;
  }

  // ════════════════════════════════════════════════════════════════════
  //  DOWNLOAD / FULLSCREEN / PROGRESS / TOAST
  // ════════════════════════════════════════════════════════════════════
  function downloadPNG() {
    if (!currentImageURL) { toast('Nothing rendered yet', true); return; }
    const a = document.createElement('a');
    a.href = currentImageURL;
    a.download = `fractal_${state.fractalType}_${Date.now()}.png`;
    a.click(); toast('PNG saved!');
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
    else document.exitFullscreen();
  }

  function showLoading(msg) {
    $('loading-overlay').style.display = 'flex';
    $('loading-text').textContent = msg || 'Rendering…';
  }
  function hideLoading() { $('loading-overlay').style.display = 'none'; }

  function setProgress(pct) { progressBar.style.width = pct + '%'; }

  let toastTimer = null;
  function toast(msg, isError = false) {
    const el = $('toast');
    el.textContent = msg;
    el.className = 'show' + (isError ? ' error' : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = ''; }, 3200);
  }

  // ════════════════════════════════════════════════════════════════════
  //  KEYBOARD SHORTCUTS
  // ════════════════════════════════════════════════════════════════════
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'Enter' || e.key === 'r') render2D();
    if (e.key === 'z' || e.key === 'u')     undoZoom();
    if (e.key === 'b')                      bookmarkView();
    if (e.key === 'f')                      toggleFullscreen();
    if (e.key === 's')                      downloadPNG();
    if (e.key === '1') switchTab('2d');
    if (e.key === '2') switchTab('3d');
    if (e.key === '3') switchTab('animate');
    if (e.key === '4') switchTab('julia-grid');
    if (e.key === '5') switchTab('ifs');
  });

  // ── Boot ────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);
})();
