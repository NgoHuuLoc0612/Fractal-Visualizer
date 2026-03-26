# Fractal Visualizer — Enterprise Edition

A high-performance fractal exploration platform combining a **C++ computation engine** (via pybind11), a **Python/Flask** API server, and a **WebGL + Three.js** frontend.

---

## Architecture

```
fractal_visualizer/
├── src/
│   └── fractal_engine.cpp   # C++ core: all math, multi-threaded, pybind11 bindings
├── static/
│   ├── css/app.css          # Dark cyber UI
│   └── js/
│       ├── app.js           # Frontend controller (2D, anim, grid, IFS, bookmarks)
│       └── three-surface.js # Three.js 3D height-map renderer
├── templates/
│   └── index.html           # Single-page app shell
├── server.py                # Flask API (render, heightmap, julia_grid, animation_frame)
├── setup.py                 # pybind11 build script
└── requirements.txt
```

---

## Supported Fractals

| Type         | Features                                    |
|--------------|---------------------------------------------|
| Mandelbrot   | Smooth, orbit-trap, distance, angle, stripe |
| Julia        | Full c-parameter sweep, animation           |
| Burning Ship | Absolute-value variant                      |
| Newton       | Degree 2–8, convergence colouring           |
| Lyapunov     | Custom AB sequence, warmup/iter settings    |
| IFS          | Barnsley, Sierpinski, Dragon, Lévy, Tree    |

## Coloring Algorithms

- **Smooth Iteration** – continuous escape-time with log normalisation  
- **Orbit Trap** – distance to a point in the complex plane  
- **Distance Estimate** – exterior distance to the boundary  
- **Argument Angle** – phase of the final iterate  
- **Stripe Average** – interference-pattern coloring  

## Palettes

`ultra` · `fire` · `ice` · `electric` · `grayscale` · `newton`

---

## Build & Run

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Compile the C++ extension

```bash
python setup.py build_ext --inplace
```

> Requires a C++17 compiler (GCC ≥ 9, Clang ≥ 10, MSVC 2019+).  
> On macOS with Homebrew: `brew install llvm libomp` if OpenMP is needed.

### 3. Start the server

```bash
python server.py
# or with debug mode:
DEBUG=1 python server.py
```

Open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Path                      | Description                            |
|--------|---------------------------|----------------------------------------|
| GET    | `/api/info`               | Engine info, thread count, feature list|
| POST   | `/api/render`             | Render fractal → PNG base64            |
| POST   | `/api/heightmap`          | Float32 height array for 3D surface    |
| POST   | `/api/julia_preview_grid` | 20-tile Julia parameter grid           |
| POST   | `/api/animation_frame`    | Single Julia animation frame           |

### Example render request

```json
{
  "fractal_type": "mandelbrot",
  "xmin": -2.5, "xmax": 1.0, "ymin": -1.25, "ymax": 1.25,
  "width": 1200, "height": 900,
  "max_iter": 512,
  "palette": "ultra",
  "color_mode": 0,
  "gamma": 1.2,
  "cycle_speed": 1.0,
  "bailout": 2.0,
  "num_threads": 8
}
```

---

## Frontend Features

- **Drag-to-zoom** with real-time rubber-band selection  
- **Scroll-wheel** zoom centred on cursor  
- **Zoom history** stack with undo  
- **Bookmarks** persisted to localStorage  
- **Shareable URLs** encoding the full view state  
- **3D Surface** — Three.js terrain with orbit controls, vertex colouring, wireframe  
- **Julia Animation** — continuous parameter sweep at target FPS  
- **Julia Grid** — 20 famous Julia sets rendered in parallel  
- **IFS Explorer** — 5 preset iterated function systems  
- **Keyboard shortcuts**: `Enter`=render, `z`=undo, `b`=bookmark, `s`=save, `f`=fullscreen, `1–5`=tabs  

---

## Performance Notes

- The C++ engine uses `std::async` with configurable thread count (up to hardware concurrency).  
- All fractals are computed tile-row-parallel; no OpenMP dependency (pure C++11 threads).  
- A Python fallback (NumPy vectorised) is used automatically if the C++ module is not built.  
- For 4K renders (3840×2160) at 1024 iterations, expect ~2–5 s on an 8-core desktop with the native engine.
