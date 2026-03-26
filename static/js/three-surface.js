/**
 * three-surface.js
 * Renders fractal data as a 3D height-map terrain using Three.js r128
 */
window.FractalSurface = (function () {
  'use strict';

  let renderer, scene, camera, mesh, controls, animId;
  let container, initialized = false;

  // ── simple orbit controls ─────────────────────────────────────────────
  function makeOrbitControls(camera, domEl) {
    let isDragging = false, isRight = false;
    let lastX = 0, lastY = 0;
    let phi = Math.PI / 3, theta = Math.PI / 4, radius = 3.5;
    let tx = 0, ty = 0;

    function update() {
      camera.position.set(
        tx + radius * Math.sin(phi) * Math.cos(theta),
        ty + radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta)
      );
      camera.lookAt(tx, ty, 0);
    }
    update();

    domEl.addEventListener('mousedown', e => {
      isDragging = true; isRight = e.button === 2;
      lastX = e.clientX; lastY = e.clientY;
      e.preventDefault();
    });
    window.addEventListener('mouseup', () => { isDragging = false; });
    window.addEventListener('mousemove', e => {
      if (!isDragging) return;
      let dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (isRight) {
        tx -= dx * 0.003; ty += dy * 0.003;
      } else {
        theta -= dx * 0.005; phi   = Math.max(0.05, Math.min(Math.PI - 0.05, phi + dy * 0.005));
      }
      update();
    });
    domEl.addEventListener('wheel', e => {
      radius = Math.max(0.5, Math.min(12, radius + e.deltaY * 0.005));
      update(); e.preventDefault();
    }, { passive: false });
    domEl.addEventListener('contextmenu', e => e.preventDefault());
    return { update };
  }

  // ── init Three.js ─────────────────────────────────────────────────────
  function init(containerId) {
    container = document.getElementById(containerId);
    if (!container || initialized) return;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x060810, 1);
    container.appendChild(renderer.domElement);
    resize();

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x060810, 0.08);

    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 100);
    camera.position.set(0, 2, 3.5);

    // Lights
    const ambient = new THREE.AmbientLight(0x223355, 0.5);
    scene.add(ambient);
    const dir1 = new THREE.DirectionalLight(0x00ffe7, 1.2);
    dir1.position.set(2, 4, 2);
    scene.add(dir1);
    const dir2 = new THREE.DirectionalLight(0xff3cac, 0.7);
    dir2.position.set(-3, 2, -1);
    scene.add(dir2);

    controls = makeOrbitControls(camera, renderer.domElement);

    window.addEventListener('resize', resize);
    initialized = true;
    animLoop();
  }

  function resize() {
    if (!container || !renderer) return;
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h);
    if (camera) { camera.aspect = w / h; camera.updateProjectionMatrix(); }
  }

  function animLoop() {
    animId = requestAnimationFrame(animLoop);
    renderer.render(scene, camera);
  }

  // ── build mesh from Float32 height data ──────────────────────────────
  function buildSurface(float32b64, rows, cols, wireframe) {
    if (!initialized) return;

    // Remove old mesh
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); mesh = null; }

    const raw = Uint8Array.from(atob(float32b64), c => c.charCodeAt(0));
    const heights = new Float32Array(raw.buffer);

    const geo = new THREE.PlaneGeometry(2, 2, cols - 1, rows - 1);
    geo.rotateX(-Math.PI / 2);

    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      pos.setY(i, heights[i] || 0);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();

    // Vertex colours from height
    const colors = new Float32Array(pos.count * 3);
    let hmin = Infinity, hmax = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const h = pos.getY(i);
      if (h < hmin) hmin = h;
      if (h > hmax) hmax = h;
    }
    const palStops = [
      [0.00, [0.024, 0.004, 0.184]],
      [0.25, [0.000, 0.392, 0.820]],
      [0.55, [0.525, 0.710, 0.898]],
      [0.80, [0.973, 0.788, 0.373]],
      [1.00, [1.000, 1.000, 1.000]],
    ];
    function palLerp(t) {
      let i = 0;
      while (i < palStops.length - 2 && palStops[i + 1][0] < t) i++;
      const [t0, c0] = palStops[i], [t1, c1] = palStops[i + 1];
      const f = (t - t0) / (t1 - t0 + 1e-12);
      return c0.map((v, k) => v + (c1[k] - v) * f);
    }
    for (let i = 0; i < pos.count; i++) {
      const t = (pos.getY(i) - hmin) / (hmax - hmin + 1e-9);
      const [r, g, b] = palLerp(Math.pow(t, 0.7));
      colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = wireframe
      ? new THREE.MeshBasicMaterial({ vertexColors: true, wireframe: true })
      : new THREE.MeshPhongMaterial({
          vertexColors: true, shininess: 60,
          specular: new THREE.Color(0x00ffe7),
          side: THREE.DoubleSide
        });

    mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
  }

  // ── public API ────────────────────────────────────────────────────────
  return { init, buildSurface, resize };
})();
