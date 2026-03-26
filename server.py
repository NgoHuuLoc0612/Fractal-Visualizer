"""
Fractal Visualizer — Python/Flask backend
Bridges the C++ fractal_core extension with the web frontend.

Supported fractal types:
    mandelbrot, julia, burning_ship, newton, lyapunov, ifs,
    tricorn, multibrot, phoenix, burning_ship_julia,
    nova, collatz, buddhabrot
"""

from __future__ import annotations
import base64, io, logging, math, os, sys, time
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

# ── C++ extension ────────────────────────────────────────────────────────────
try:
    import fractal_core as _fc
    NATIVE = True
except ImportError:
    NATIVE = False
    logging.warning("fractal_core not found — using Python fallback")

# ─────────────────────────────────────────────────────────────────────────────
#  Pure-Python fallbacks  (slow but always available)
# ─────────────────────────────────────────────────────────────────────────────
def _py_mandelbrot(xmin,xmax,ymin,ymax,width,height,max_iter,**_):
    x=np.linspace(xmin,xmax,width); y=np.linspace(ymin,ymax,height)
    C=x[np.newaxis,:]+1j*y[:,np.newaxis]; Z=np.zeros_like(C); M=np.zeros(C.shape,dtype=float)
    for i in range(max_iter):
        mask=np.abs(Z)<=2; Z[mask]=Z[mask]**2+C[mask]
        M[mask&(np.abs(Z)>2)]=i+1
    return M

def _py_julia(xmin,xmax,ymin,ymax,width,height,max_iter,cr,ci,**_):
    x=np.linspace(xmin,xmax,width); y=np.linspace(ymin,ymax,height)
    Z=x[np.newaxis,:]+1j*y[:,np.newaxis]; C=complex(cr,ci); M=np.zeros(Z.shape,dtype=float)
    for i in range(max_iter):
        mask=np.abs(Z)<=2; Z[mask]=Z[mask]**2+C
        M[mask&(np.abs(Z)>2)]=i+1
    return M

def _py_tricorn(xmin,xmax,ymin,ymax,width,height,max_iter,**_):
    """Mandelbar: z_{n+1}=conj(z)^2+c"""
    x=np.linspace(xmin,xmax,width); y=np.linspace(ymin,ymax,height)
    C=x[np.newaxis,:]+1j*y[:,np.newaxis]; Z=np.zeros_like(C); M=np.zeros(C.shape,dtype=float)
    for i in range(max_iter):
        mask=np.abs(Z)<=2; Z[mask]=np.conj(Z[mask])**2+C[mask]
        M[mask&(np.abs(Z)>2)]=i+1
    return M

def _py_multibrot(xmin,xmax,ymin,ymax,width,height,max_iter,exponent=3.0,**_):
    x=np.linspace(xmin,xmax,width); y=np.linspace(ymin,ymax,height)
    C=x[np.newaxis,:]+1j*y[:,np.newaxis]; Z=np.zeros_like(C); M=np.zeros(C.shape,dtype=float)
    for i in range(max_iter):
        mask=np.abs(Z)<=2; Z[mask]=Z[mask]**exponent+C[mask]
        M[mask&(np.abs(Z)>2)]=i+1
    return M

def _py_phoenix(xmin,xmax,ymin,ymax,width,height,max_iter,cr=0.5667,ci=-0.5,**_):
    x=np.linspace(xmin,xmax,width); y=np.linspace(ymin,ymax,height)
    Z=x[np.newaxis,:]+1j*y[:,np.newaxis]; prev=np.zeros_like(Z); M=np.zeros(Z.shape,dtype=float)
    for i in range(max_iter):
        mask=np.abs(Z)<=2
        nz=Z[mask]**2+cr+ci*prev[mask]; prev[mask]=Z[mask]; Z[mask]=nz
        M[mask&(np.abs(Z)>2)]=i+1
    return M

def _py_colormap(raw,palette="ultra",gamma=1.0,invert=False,**_):
    vmin,vmax=raw.min(),raw.max()
    t=(raw-vmin)/(vmax-vmin+1e-12)
    if invert: t=1-t
    t=np.power(np.clip(t,0,1),gamma)
    palettes={
        "ultra":[(0,0,0),(9,1,47),(25,7,107),(134,181,229),(211,236,248),(248,201,95),(0,0,0)],
        "fire":[(0,0,0),(128,0,0),(255,64,0),(255,200,0),(255,255,200)],
        "ice":[(0,0,32),(0,64,128),(64,196,255),(200,240,255),(255,255,255)],
        "electric":[(10,0,30),(60,0,200),(0,100,255),(0,255,220),(255,255,255)],
        "grayscale":[(0,0,0),(255,255,255)],
        "magma":[(0,0,4),(79,18,123),(181,54,122),(251,135,97),(252,253,191)],
        "viridis":[(68,1,84),(59,82,139),(33,145,140),(94,201,98),(253,231,37)],
        "plasma":[(13,8,135),(126,3,168),(204,71,120),(248,149,64),(240,249,33)],
        "cosmic":[(0,0,0),(80,0,120),(180,20,200),(255,100,255),(255,220,255)],
        "tropical":[(0,30,60),(0,120,200),(0,200,150),(255,200,0),(255,100,0)],
    }
    stops=np.array(palettes.get(palette,palettes["ultra"]),dtype=np.float32)
    n=len(stops)
    idx=np.clip(t,0,1)*(n-1)
    i0=np.floor(idx).astype(int); i1=np.minimum(i0+1,n-1); f=idx-i0
    r=(stops[i0,0]+(stops[i1,0]-stops[i0,0])*f).astype(np.uint8)
    g=(stops[i0,1]+(stops[i1,1]-stops[i0,1])*f).astype(np.uint8)
    b=(stops[i0,2]+(stops[i1,2]-stops[i0,2])*f).astype(np.uint8)
    return np.stack([r,g,b,np.full_like(r,255)],axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Default viewport coords per fractal type
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_VIEWS: dict[str, dict] = {
    "mandelbrot":          {"xmin":-2.5,"xmax":1.0,"ymin":-1.25,"ymax":1.25},
    "julia":               {"xmin":-2.0,"xmax":2.0,"ymin":-1.5,"ymax":1.5},
    "burning_ship":        {"xmin":-2.5,"xmax":1.5,"ymin":-2.0,"ymax":0.5},
    "newton":              {"xmin":-2.0,"xmax":2.0,"ymin":-2.0,"ymax":2.0},
    "lyapunov":            {"xmin":2.0,"xmax":4.0,"ymin":2.0,"ymax":4.0},
    "tricorn":             {"xmin":-2.5,"xmax":1.0,"ymin":-1.25,"ymax":1.25},
    "multibrot":           {"xmin":-2.0,"xmax":2.0,"ymin":-2.0,"ymax":2.0},
    "phoenix":             {"xmin":-2.0,"xmax":2.0,"ymin":-2.0,"ymax":2.0},
    "burning_ship_julia":  {"xmin":-2.0,"xmax":2.0,"ymin":-2.0,"ymax":2.0},
    "nova":                {"xmin":-2.0,"xmax":2.0,"ymin":-2.0,"ymax":2.0},
    "collatz":             {"xmin":-3.0,"xmax":3.0,"ymin":-3.0,"ymax":3.0},
    "buddhabrot":          {"xmin":-2.5,"xmax":1.0,"ymin":-1.25,"ymax":1.25},
    "ifs":                 {"xmin":-3.0,"xmax":3.0,"ymin":-1.0,"ymax":10.0},
}

IFS_PRESETS = {
    "barnsley":   [[0.0,0.0,0.0,0.16,0.0,0.0,0.01],[0.85,0.04,-0.04,0.85,0.0,1.60,0.85],[0.20,-0.26,0.23,0.22,0.0,1.60,0.07],[-0.15,0.28,0.26,0.24,0.0,0.44,0.07]],
    "sierpinski": [[0.5,0.0,0.0,0.5,0.0,0.0,0.33],[0.5,0.0,0.0,0.5,0.5,0.0,0.33],[0.5,0.0,0.0,0.5,0.25,0.5,0.34]],
    "dragon":     [[0.824074,0.281482,-0.212346,0.864198,-1.882290,-0.110607,0.787473],[0.088272,0.520988,-0.463889,-0.377778,0.785360,8.095795,0.212527]],
    "levy":       [[0.5,-0.5,0.5,0.5,0.0,0.0,0.5],[0.5,0.5,-0.5,0.5,0.5,0.5,0.5]],
    "tree":       [[0.0,0.0,0.0,0.5,0.0,0.0,0.05],[0.42,-0.42,0.42,0.42,0.0,0.2,0.4],[0.42,0.42,-0.42,0.42,0.0,0.2,0.4],[0.1,0.0,0.0,0.1,0.0,0.2,0.15]],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Core dispatch
# ─────────────────────────────────────────────────────────────────────────────
def compute_fractal(params: dict) -> np.ndarray:
    ftype  = params.get("fractal_type","mandelbrot")
    dv     = DEFAULT_VIEWS.get(ftype, DEFAULT_VIEWS["mandelbrot"])
    w      = int(params.get("width",  800))
    h      = int(params.get("height", 600))
    mi     = int(params.get("max_iter", 256))
    nt     = int(params.get("num_threads", 4))
    bail   = float(params.get("bailout", 2.0))
    xmin   = float(params.get("xmin", dv["xmin"]))
    xmax   = float(params.get("xmax", dv["xmax"]))
    ymin   = float(params.get("ymin", dv["ymin"]))
    ymax   = float(params.get("ymax", dv["ymax"]))
    cm     = int(params.get("color_mode", 0))
    stripe = float(params.get("stripe_density", 5.0))
    otx    = float(params.get("orbit_trap_x", 0.0))
    oty    = float(params.get("orbit_trap_y", 0.0))

    # ── classic types ─────────────────────────────────────────────────────────
    if ftype == "mandelbrot":
        if NATIVE:
            return _fc.compute_mandelbrot(xmin,xmax,ymin,ymax,w,h,mi,cm,bail,stripe,otx,oty,nt)
        return _py_mandelbrot(xmin,xmax,ymin,ymax,w,h,mi)

    if ftype == "julia":
        cr=float(params.get("julia_cr",-0.7)); ci=float(params.get("julia_ci",0.27015))
        if NATIVE:
            return _fc.compute_julia(xmin,xmax,ymin,ymax,w,h,mi,cr,ci,cm,bail,nt)
        return _py_julia(xmin,xmax,ymin,ymax,w,h,mi,cr,ci)

    if ftype == "burning_ship":
        if NATIVE:
            return _fc.compute_burning_ship(xmin,xmax,ymin,ymax,w,h,mi,bail,nt)
        return _py_mandelbrot(xmin,xmax,ymin,ymax,w,h,mi)

    if ftype == "newton":
        deg=int(params.get("newton_degree",3)); tol=float(params.get("newton_tol",1e-6))
        if NATIVE:
            return _fc.compute_newton(xmin,xmax,ymin,ymax,w,h,mi,deg,tol,nt)
        return _py_mandelbrot(xmin,xmax,ymin,ymax,w,h,mi)

    if ftype == "lyapunov":
        seq=params.get("lyapunov_seq","AB")
        wu=int(params.get("lyapunov_warmup",100)); it=int(params.get("lyapunov_iters",200))
        if NATIVE:
            return _fc.compute_lyapunov(xmin,xmax,ymin,ymax,w,h,seq,wu,it,nt)
        return _py_mandelbrot(xmin,xmax,ymin,ymax,w,h,mi)

    if ftype == "ifs":
        preset=params.get("ifs_preset","barnsley")
        transforms=params.get("ifs_transforms", IFS_PRESETS.get(preset, IFS_PRESETS["barnsley"]))
        np_=int(params.get("ifs_points",500_000)); pal=params.get("palette","green")
        if NATIVE:
            return _fc.compute_ifs(w,h,np_,transforms,pal)
        return np.zeros((h,w,4),dtype=np.uint8)

    # ── new types ─────────────────────────────────────────────────────────────
    if ftype == "tricorn":
        if NATIVE:
            return _fc.compute_tricorn(xmin,xmax,ymin,ymax,w,h,mi,bail,nt)
        return _py_tricorn(xmin,xmax,ymin,ymax,w,h,mi)

    if ftype == "multibrot":
        exp=float(params.get("multibrot_exp",3.0))
        if NATIVE:
            return _fc.compute_multibrot(xmin,xmax,ymin,ymax,w,h,mi,exp,bail,nt)
        return _py_multibrot(xmin,xmax,ymin,ymax,w,h,mi,exponent=exp)

    if ftype == "phoenix":
        cr=float(params.get("phoenix_cr",0.5667)); ci=float(params.get("phoenix_ci",-0.5))
        if NATIVE:
            return _fc.compute_phoenix(xmin,xmax,ymin,ymax,w,h,mi,cr,ci,bail,nt)
        return _py_phoenix(xmin,xmax,ymin,ymax,w,h,mi,cr,ci)

    if ftype == "burning_ship_julia":
        cr=float(params.get("bsj_cr",-1.755)); ci=float(params.get("bsj_ci",0.0))
        if NATIVE:
            return _fc.compute_burning_ship_julia(xmin,xmax,ymin,ymax,w,h,mi,cr,ci,bail,nt)
        return _py_mandelbrot(xmin,xmax,ymin,ymax,w,h,mi)

    if ftype == "nova":
        deg=int(params.get("nova_degree",3)); tol=float(params.get("nova_tol",1e-6))
        if NATIVE:
            return _fc.compute_nova(xmin,xmax,ymin,ymax,w,h,mi,deg,tol,bail,nt)
        return _py_mandelbrot(xmin,xmax,ymin,ymax,w,h,mi)

    if ftype == "collatz":
        if NATIVE:
            return _fc.compute_collatz(xmin,xmax,ymin,ymax,w,h,mi,
                                       float(params.get("bailout",32.0)),nt)
        return _py_mandelbrot(xmin,xmax,ymin,ymax,w,h,mi)

    if ftype == "buddhabrot":
        mir=int(params.get("brot_iter_r",100)); mig=int(params.get("brot_iter_g",1000))
        mib=int(params.get("brot_iter_b",5000)); ns=int(params.get("brot_samples",2_000_000))
        if NATIVE:
            return _fc.compute_buddhabrot(xmin,xmax,ymin,ymax,w,h,mir,mig,mib,ns,nt)
        return np.zeros((h,w,4),dtype=np.uint8)

    raise ValueError(f"Unknown fractal type: {ftype!r}")


def raw_to_rgba(raw:np.ndarray, params:dict) -> np.ndarray:
    if raw.ndim==3: return raw   # already RGBA (IFS, Buddhabrot)
    pal=params.get("palette","ultra"); gamma=float(params.get("gamma",1.0))
    invert=bool(params.get("invert",False)); speed=float(params.get("cycle_speed",1.0))
    if NATIVE:
        return _fc.apply_colormap(raw,pal,gamma,invert,True,speed)
    return _py_colormap(raw,pal,gamma,invert)

def rgba_to_png_b64(rgba:np.ndarray) -> str:
    img=Image.fromarray(rgba.astype(np.uint8),mode="RGBA")
    buf=io.BytesIO(); img.save(buf,format="PNG",optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
#  Flask app
# ─────────────────────────────────────────────────────────────────────────────
app=Flask(__name__,
    static_folder=os.path.join(os.path.dirname(__file__),"static"),
    template_folder=os.path.join(os.path.dirname(__file__),"templates"))
CORS(app)
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
log=logging.getLogger(__name__)


@app.route("/")
def index():
    return send_from_directory(app.template_folder,"index.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder,path)


@app.route("/api/info",methods=["GET"])
def api_info():
    return jsonify({
        "native_engine": NATIVE,
        "hardware_threads": _fc.get_hardware_threads() if NATIVE else 1,
        "fractals": list(DEFAULT_VIEWS.keys()),
        "ifs_presets": list(IFS_PRESETS.keys()),
        "palettes": ["ultra","fire","ice","electric","grayscale","newton",
                     "magma","viridis","plasma","cosmic","tropical"],
        "color_modes": {"0":"smooth","1":"orbit trap","2":"distance est.",
                        "3":"argument angle","4":"stripe average"},
        "default_views": DEFAULT_VIEWS,
    })


@app.route("/api/render",methods=["POST"])
def api_render():
    params=request.get_json(force=True,silent=True) or {}
    t0=time.perf_counter()
    try:
        raw=compute_fractal(params)
        rgba=raw_to_rgba(raw,params)
        b64=rgba_to_png_b64(rgba)
        return jsonify({"ok":True,"image":f"data:image/png;base64,{b64}",
                        "render_ms":round((time.perf_counter()-t0)*1000,1),
                        "width":rgba.shape[1],"height":rgba.shape[0]})
    except Exception as exc:
        log.exception("Render error"); return jsonify({"ok":False,"error":str(exc)}),500


@app.route("/api/heightmap",methods=["POST"])
def api_heightmap():
    params=request.get_json(force=True,silent=True) or {}
    try:
        raw=compute_fractal(params); z_scale=float(params.get("z_scale",1.0))
        smooth=bool(params.get("smooth_3d",True))
        if raw.ndim==3: raw=raw[:,:,0].astype(float)
        if NATIVE:
            hm=_fc.compute_heightmap(raw,z_scale,smooth)
        else:
            vmin,vmax=raw.min(),raw.max()
            hm=((raw-vmin)/(vmax-vmin+1e-12)*z_scale).astype(np.float32)
        flat=hm.astype(np.float32).flatten()
        b64=base64.b64encode(flat.tobytes()).decode()
        return jsonify({"ok":True,"data":b64,"rows":hm.shape[0],"cols":hm.shape[1]})
    except Exception as exc:
        log.exception("Heightmap error"); return jsonify({"ok":False,"error":str(exc)}),500


@app.route("/api/julia_preview_grid",methods=["POST"])
def julia_preview_grid():
    params=request.get_json(force=True,silent=True) or {}
    tw,th=160,120; mi=int(params.get("max_iter",128))
    bail=float(params.get("bailout",2.0)); pal=params.get("palette","ultra")
    nt=int(params.get("num_threads",2))
    seeds=[(-0.7,0.27015),(-0.4,0.6),(0.285,0.01),(-0.835,-0.2321),(0.45,0.1428),
           (-0.70176,-0.3842),(0.0,0.8),(-0.123,0.745),(0.3,-0.5),(-0.5,0.5),
           (0.285,0.013),(0.7,0.2),(-0.4,-0.59),(0.34,-0.05),(-0.75,0.11),
           (0.0,-0.8),(0.15,1.04),(-0.625,0.0),(-0.1,0.651),(0.0,0.0)]
    images=[]
    for cr,ci in seeds:
        if NATIVE:
            raw=_fc.compute_julia(-1.5,1.5,-1.5,1.5,tw,th,mi,cr,ci,0,bail,nt)
            rgba=_fc.apply_colormap(raw,pal,1.0,False,True,1.0)
        else:
            raw=_py_julia(-1.5,1.5,-1.5,1.5,tw,th,mi,cr,ci)
            rgba=_py_colormap(raw,pal)
        images.append({"cr":cr,"ci":ci,"image":f"data:image/png;base64,{rgba_to_png_b64(rgba)}"})
    return jsonify({"ok":True,"images":images})


@app.route("/api/animation_frame",methods=["POST"])
def animation_frame():
    params=request.get_json(force=True,silent=True) or {}
    angle=float(params.get("angle",0.0)); radius=float(params.get("radius",0.7885))
    params["julia_cr"]=radius*math.cos(angle); params["julia_ci"]=radius*math.sin(angle)
    params["fractal_type"]="julia"
    t0=time.perf_counter()
    raw=compute_fractal(params); rgba=raw_to_rgba(raw,params); b64=rgba_to_png_b64(rgba)
    return jsonify({"ok":True,"image":f"data:image/png;base64,{b64}",
                    "render_ms":round((time.perf_counter()-t0)*1000,1),
                    "cr":params["julia_cr"],"ci":params["julia_ci"]})


@app.route("/api/default_view",methods=["GET"])
def api_default_view():
    ftype=request.args.get("type","mandelbrot")
    return jsonify(DEFAULT_VIEWS.get(ftype, DEFAULT_VIEWS["mandelbrot"]))


if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    debug=os.environ.get("DEBUG","0")=="1"
    log.info("Fractal Visualizer on port %d (native=%s)",port,NATIVE)
    app.run(host="0.0.0.0",port=port,debug=debug,threaded=True)
