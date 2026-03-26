/*
 * fractal_engine.cpp  —  pybind11 C++ core
 *
 * Fractal types:
 *   mandelbrot, julia, burning_ship, newton, lyapunov, ifs,
 *   tricorn, multibrot, phoenix, burning_ship_julia,
 *   nova, collatz, buddhabrot
 *
 * Build (MSVC):      python setup.py build_ext --inplace
 * Build (GCC/MinGW): same — setup.py auto-selects flags
 */

#ifdef _MSC_VER
#  define _USE_MATH_DEFINES   // M_PI, M_E, ... in <cmath> for MSVC
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>
#include <random>
#include <numeric>
#include <atomic>
#include <string>

namespace py = pybind11;
using complex_d = std::complex<double>;
using RGB = std::tuple<uint8_t,uint8_t,uint8_t>;

// ─────────────────────────────────────────────────────────────
//  Utility
// ─────────────────────────────────────────────────────────────
template<typename T>
inline T mclamp(T v, T lo, T hi){ return v<lo?lo:v>hi?hi:v; }

struct ColoringParams {
    int    color_mode;
    double bailout;
    double stripe_density;
    double orbit_trap_x;
    double orbit_trap_y;
};

inline double smooth_iter(int iter, int max_iter, complex_d z, double bail) {
    if (iter==max_iter) return 0.0;
    return iter+1.0 - std::log(std::log(std::abs(z))/std::log(bail))/std::log(2.0);
}
inline double orbit_trap_val(int iter, int max_iter, complex_d z, double tx, double ty) {
    if (iter==max_iter) return 0.0;
    double dx=z.real()-tx, dy=z.imag()-ty;
    return std::sqrt(dx*dx+dy*dy);
}
inline double stripe_avg(complex_d z, double density) {
    return 0.5*std::sin(density*std::arg(z))+0.5;
}

// Thread pool: split rows [0,height) into nt bands
template<typename F>
void parallel_rows(int height, int nt_req, F&& fn){
    int nt=std::max(1,std::min(nt_req,(int)std::thread::hardware_concurrency()));
    std::vector<std::future<void>> futs;
    futs.reserve(nt);
    int chunk=height/nt;
    for(int t=0;t<nt;++t){
        int rs=t*chunk, re=(t==nt-1)?height:rs+chunk;
        futs.push_back(std::async(std::launch::async,fn,rs,re));
    }
    for(auto& f:futs) f.get();
}

// ═════════════════════════════════════════════════════════════
//  1. MANDELBROT
// ═════════════════════════════════════════════════════════════
static double mandelbrot_pixel(double cr, double ci,
                               int max_iter, const ColoringParams& cp){
    double zr=0,zi=0,zr2=0,zi2=0,stripe_sum=0;
    int iter=0;
    double bail2=cp.bailout*cp.bailout;
    while(iter<max_iter && zr2+zi2<bail2){
        zi=2.0*zr*zi+ci; zr=zr2-zi2+cr;
        zr2=zr*zr; zi2=zi*zi;
        if(cp.color_mode==4) stripe_sum+=stripe_avg({zr,zi},cp.stripe_density);
        ++iter;
    }
    if(cp.color_mode==0) return smooth_iter(iter,max_iter,{zr,zi},cp.bailout);
    if(cp.color_mode==1) return orbit_trap_val(iter,max_iter,{zr,zi},cp.orbit_trap_x,cp.orbit_trap_y);
    if(cp.color_mode==3) return (iter==max_iter)?0.0:(std::atan2(zi,zr)/M_PI+1.0)*0.5;
    if(cp.color_mode==4) return (iter==max_iter||iter==0)?0.0:stripe_sum/iter;
    if(iter==max_iter) return 0.0;
    double mod=std::sqrt(zr2+zi2); return mod*std::log(mod);
}

py::array_t<double> compute_mandelbrot(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,
    int color_mode,double bailout,
    double stripe_density,double orbit_trap_x,double orbit_trap_y,
    int num_threads)
{
    ColoringParams cp{color_mode,bailout,stripe_density,orbit_trap_x,orbit_trap_y};
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width, dy=(ymax-ymin)/height;
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double ci=ymin+r*dy;
            for(int c=0;c<width;++c) buf(r,c)=mandelbrot_pixel(xmin+c*dx,ci,max_iter,cp);}
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  2. JULIA
// ═════════════════════════════════════════════════════════════
static double julia_pixel(double zr,double zi,double cr,double ci,
                          int max_iter,const ColoringParams& cp){
    double zr2=zr*zr,zi2=zi*zi,bail2=cp.bailout*cp.bailout;
    int iter=0;
    while(iter<max_iter&&zr2+zi2<bail2){
        zi=2.0*zr*zi+ci; zr=zr2-zi2+cr;
        zr2=zr*zr; zi2=zi*zi; ++iter;
    }
    return smooth_iter(iter,max_iter,{zr,zi},cp.bailout);
}

py::array_t<double> compute_julia(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,
    double cr,double ci,int color_mode,double bailout,int num_threads)
{
    ColoringParams cp{color_mode,bailout,5.0,0.0,0.0};
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height;
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double zi=ymin+r*dy;
            for(int c=0;c<width;++c) buf(r,c)=julia_pixel(xmin+c*dx,zi,cr,ci,max_iter,cp);}
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  3. BURNING SHIP
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_burning_ship(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,double bailout,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height,bail2=bailout*bailout;
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double ci=ymin+r*dy;
            for(int c=0;c<width;++c){
                double cr=xmin+c*dx,zr=0,zi=0; int iter=0;
                while(iter<max_iter&&zr*zr+zi*zi<bail2){
                    double tmp=zr*zr-zi*zi+cr;
                    zi=std::abs(2.0*zr*zi)+ci; zr=tmp; ++iter;
                }
                buf(r,c)=(iter==max_iter)?0.0:iter+1.0-std::log(std::log(std::sqrt(zr*zr+zi*zi))/std::log(bailout))/std::log(2.0);
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  4. NEWTON
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_newton(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,
    int poly_degree,double tol,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height;
    int n=std::max(2,std::min(poly_degree,8));
    std::vector<complex_d> roots(n);
    for(int k=0;k<n;++k) roots[k]=std::polar(1.0,2.0*M_PI*k/n);
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double yi=ymin+r*dy;
            for(int c=0;c<width;++c){
                complex_d z(xmin+c*dx,yi); int iter=0;
                while(iter<max_iter){
                    complex_d zp1=std::pow(z,n-1),zp=zp1*z;
                    z-=(zp-complex_d(1,0))/(complex_d(n,0)*zp1);
                    bool cv=false; for(auto&ro:roots)if(std::abs(z-ro)<tol){cv=true;break;}
                    if(cv) break; ++iter;
                }
                int ri=0; double best=1e18;
                for(int k=0;k<n;++k){double d=std::abs(z-roots[k]);if(d<best){best=d;ri=k;}}
                buf(r,c)=ri+(double)iter/max_iter;
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  5. LYAPUNOV
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_lyapunov(
    double amin,double amax,double bmin,double bmax,
    int width,int height,
    const std::string& sequence,int warmup,int iters,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double da=(amax-amin)/width,db=(bmax-bmin)/height;
    int slen=(int)sequence.size();
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double b=bmin+r*db;
            for(int c=0;c<width;++c){
                double a=amin+c*da,x=0.5;
                for(int i=0;i<warmup;++i){double rv=(sequence[i%slen]=='A')?a:b;x=rv*x*(1.0-x);}
                double lyap=0.0;
                for(int i=0;i<iters;++i){
                    double rv=(sequence[i%slen]=='A')?a:b;
                    x=rv*x*(1.0-x);
                    double v=std::abs(rv*(1.0-2.0*x));
                    if(v>1e-15) lyap+=std::log(v);
                }
                buf(r,c)=lyap/iters;
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  6. TRICORN  (Mandelbar)
//     z_{n+1} = conj(z_n)^2 + c
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_tricorn(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,double bailout,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height,bail2=bailout*bailout;
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double ci=ymin+r*dy;
            for(int c=0;c<width;++c){
                double cr=xmin+c*dx,zr=0,zi=0; int iter=0;
                while(iter<max_iter&&zr*zr+zi*zi<bail2){
                    double nr=zr*zr-zi*zi+cr;
                    double ni=-2.0*zr*zi+ci;   // conjugate: negate Im part
                    zr=nr; zi=ni; ++iter;
                }
                if(iter==max_iter){buf(r,c)=0.0;continue;}
                buf(r,c)=smooth_iter(iter,max_iter,{zr,zi},bailout);
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  7. MULTIBROT  z^p + c  (real exponent generalisation)
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_multibrot(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,
    double exponent,double bailout,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height,bail2=bailout*bailout;
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double ci=ymin+r*dy;
            for(int c=0;c<width;++c){
                complex_d z(0,0),cv(xmin+c*dx,ci); int iter=0;
                while(iter<max_iter&&std::norm(z)<bail2){z=std::pow(z,exponent)+cv;++iter;}
                if(iter==max_iter){buf(r,c)=0.0;continue;}
                double log_z=std::log(std::abs(z));
                double nu=std::log(log_z/std::log(bailout))/std::log(std::abs(exponent));
                buf(r,c)=iter+1.0-nu;
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  8. PHOENIX FRACTAL
//     z_{n+1} = z_n^2 + p + q*z_{n-1}   where p=cr, q=ci
//     Pixel coords map to initial z; (p,q) are constants
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_phoenix(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,
    double cr,double ci,double bailout,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height,bail2=bailout*bailout;
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double py_=ymin+r*dy;
            for(int c=0;c<width;++c){
                double px=xmin+c*dx;
                double zr=px,zi=py_,pr=0,pi=0; int iter=0;
                while(iter<max_iter&&zr*zr+zi*zi<bail2){
                    double nr=zr*zr-zi*zi+cr+ci*pr;
                    double ni=2.0*zr*zi+ci*pi;
                    pr=zr;pi=zi; zr=nr;zi=ni; ++iter;
                }
                if(iter==max_iter){buf(r,c)=0.0;continue;}
                buf(r,c)=smooth_iter(iter,max_iter,{zr,zi},bailout);
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  9. BURNING SHIP JULIA
//     z_{n+1} = (|Re(z)| + i|Im(z)|)^2 + c_fixed
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_burning_ship_julia(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,
    double cr,double ci,double bailout,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height,bail2=bailout*bailout;
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double zi_=ymin+r*dy;
            for(int c=0;c<width;++c){
                double zr=xmin+c*dx,zi=zi_; int iter=0;
                while(iter<max_iter&&zr*zr+zi*zi<bail2){
                    double tmp=zr*zr-zi*zi+cr;
                    zi=std::abs(2.0*zr*zi)+ci; zr=tmp; ++iter;
                }
                if(iter==max_iter){buf(r,c)=0.0;continue;}
                buf(r,c)=smooth_iter(iter,max_iter,{zr,zi},bailout);
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  10. NOVA FRACTAL  (Newton + perturbation by c)
//      z_{n+1} = z_n - (z_n^p - 1)/(p*z_n^(p-1)) + c
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_nova(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,
    int poly_degree,double tol,double bailout,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height;
    int n=std::max(2,std::min(poly_degree,8));
    std::vector<complex_d> roots(n);
    for(int k=0;k<n;++k) roots[k]=std::polar(1.0,2.0*M_PI*k/n);

    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){
            for(int c=0;c<width;++c){
                complex_d cv(xmin+c*dx,ymin+r*dy);
                complex_d z(1.0,0.0); int iter=0;
                while(iter<max_iter){
                    complex_d zp1=std::pow(z,n-1);
                    complex_d step=(zp1*z-complex_d(1,0))/(complex_d(n,0)*zp1);
                    z=z-step+cv;
                    if(std::abs(step)<tol) break;
                    ++iter;
                }
                int ri=0; double best=1e18;
                for(int k=0;k<n;++k){double d=std::abs(z-roots[k]);if(d<best){best=d;ri=k;}}
                buf(r,c)=ri+(double)iter/max_iter;
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  11. COLLATZ FRACTAL  (complex extension)
//      f(z) = (1/4)(1 + 4z - (1+2z)*cos(π z))
// ═════════════════════════════════════════════════════════════
py::array_t<double> compute_collatz(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,int max_iter,double bailout,int num_threads)
{
    auto res=py::array_t<double>({height,width});
    auto buf=res.mutable_unchecked<2>();
    double dx=(xmax-xmin)/width,dy=(ymax-ymin)/height;
    parallel_rows(height,num_threads,[&](int rs,int re){
        for(int r=rs;r<re;++r){double yi=ymin+r*dy;
            for(int c=0;c<width;++c){
                complex_d z(xmin+c*dx,yi); int iter=0;
                while(iter<max_iter&&std::abs(z)<bailout){
                    complex_d cospi=std::cos(complex_d(M_PI,0)*z);
                    z=0.25*(1.0+4.0*z-(1.0+2.0*z)*cospi); ++iter;
                }
                if(iter==max_iter){buf(r,c)=0.0;continue;}
                buf(r,c)=(double)iter+1.0-std::log2(std::log(std::abs(z)+1e-12)+1e-12);
            }
        }
    });
    return res;
}

// ═════════════════════════════════════════════════════════════
//  12. BUDDHABROT  (density map of escape trajectories)
//      Returns RGBA uint8 directly (special render, no colormap step)
// ═════════════════════════════════════════════════════════════
py::array_t<uint8_t> compute_buddhabrot(
    double xmin,double xmax,double ymin,double ymax,
    int width,int height,
    int max_iter_r,int max_iter_g,int max_iter_b,
    long long num_samples,int num_threads)
{
    int sz=width*height;
    std::vector<std::atomic<int>> densR(sz),densG(sz),densB(sz);
    for(int i=0;i<sz;++i){densR[i]=0;densG[i]=0;densB[i]=0;}

    int nt=std::max(1,std::min(num_threads,(int)std::thread::hardware_concurrency()));
    long long per=num_samples/nt;
    std::vector<std::future<void>> futs;

    auto fill=[&](int seed){
        std::mt19937_64 rng((unsigned long long)seed*987654321ULL+42ULL);
        std::uniform_real_distribution<double> rx(-2.5,1.0),ry(-1.25,1.25);
        std::vector<complex_d> traj;
        traj.reserve(max_iter_b+2);
        for(long long s=0;s<per;++s){
            double cr=rx(rng),ci_=ry(rng);
            // Skip interior of main cardioid and period-2 bulb
            double q=(cr-0.25)*(cr-0.25)+ci_*ci_;
            if(q*(q+(cr-0.25))<0.25*ci_*ci_) continue;
            if((cr+1.0)*(cr+1.0)+ci_*ci_<0.0625) continue;
            complex_d cv(cr,ci_),z(0,0);
            traj.clear(); int iter=0;
            while(iter<max_iter_b&&std::norm(z)<4.0){z=z*z+cv;traj.push_back(z);++iter;}
            if(iter==max_iter_b) continue;
            bool do_r=(iter<=max_iter_r),do_g=(iter<=max_iter_g);
            for(auto& pt:traj){
                double prx=(pt.real()-xmin)/(xmax-xmin)*width;
                double pry=(pt.imag()-ymin)/(ymax-ymin)*height;
                int px=(int)prx,py=(int)pry;
                if(px<0||px>=width||py<0||py>=height) continue;
                int idx=py*width+px;
                if(do_r) densR[idx].fetch_add(1,std::memory_order_relaxed);
                if(do_g) densG[idx].fetch_add(1,std::memory_order_relaxed);
                densB[idx].fetch_add(1,std::memory_order_relaxed);
            }
        }
    };

    for(int t=0;t<nt;++t) futs.push_back(std::async(std::launch::async,fill,t));
    for(auto& f:futs) f.get();

    auto maxv=[&](const std::vector<std::atomic<int>>& d){
        int m=0;for(const auto& v:d){int vv=v.load();if(vv>m)m=vv;}return std::max(1,m);
    };
    int mxR=maxv(densR),mxG=maxv(densG),mxB=maxv(densB);

    auto out=py::array_t<uint8_t>({height,width,4});
    auto obuf=out.mutable_unchecked<3>();
    for(int r=0;r<height;++r){
        for(int c=0;c<width;++c){
            int idx=r*width+c;
            auto gm=[](double t){return (uint8_t)(std::pow(mclamp(t,0.0,1.0),0.45)*255.0);};
            obuf(r,c,0)=gm((double)densR[idx].load()/mxR);
            obuf(r,c,1)=gm((double)densG[idx].load()/mxG);
            obuf(r,c,2)=gm((double)densB[idx].load()/mxB);
            obuf(r,c,3)=255;
        }
    }
    return out;
}

// ═════════════════════════════════════════════════════════════
//  Coloring: raw float array → RGBA uint8
//  New palettes: magma, viridis, plasma, cosmic, tropical
// ═════════════════════════════════════════════════════════════
py::array_t<uint8_t> apply_colormap(
    py::array_t<double> raw,
    const std::string& palette,
    double gamma,bool invert,bool smooth_color,double cycle_speed)
{
    auto rbuf=raw.unchecked<2>();
    int H=rbuf.shape(0),W=rbuf.shape(1);
    double vmin=1e18,vmax=-1e18;
    for(int r=0;r<H;++r)for(int c=0;c<W;++c){
        double v=rbuf(r,c);if(v>vmax)vmax=v;if(v>0&&v<vmin)vmin=v;
    }
    if(vmin>=vmax) vmin=0;

    std::vector<RGB> stops;
    if(palette=="fire")
        stops={RGB{0,0,0},RGB{128,0,0},RGB{255,64,0},RGB{255,200,0},RGB{255,255,200}};
    else if(palette=="ice")
        stops={RGB{0,0,32},RGB{0,64,128},RGB{64,196,255},RGB{200,240,255},RGB{255,255,255}};
    else if(palette=="electric")
        stops={RGB{10,0,30},RGB{60,0,200},RGB{0,100,255},RGB{0,255,220},RGB{255,255,255}};
    else if(palette=="grayscale")
        stops={RGB{0,0,0},RGB{255,255,255}};
    else if(palette=="newton")
        stops={RGB{180,0,100},RGB{0,120,255},RGB{255,200,0},RGB{0,200,100},
               RGB{255,80,0},RGB{100,0,200},RGB{0,200,200},RGB{200,200,200}};
    else if(palette=="magma")
        stops={RGB{0,0,4},RGB{28,16,68},RGB{79,18,123},RGB{129,37,129},
               RGB{181,54,122},RGB{229,80,100},RGB{251,135,97},RGB{254,194,135},RGB{252,253,191}};
    else if(palette=="viridis")
        stops={RGB{68,1,84},RGB{59,82,139},RGB{33,145,140},RGB{94,201,98},RGB{253,231,37}};
    else if(palette=="plasma")
        stops={RGB{13,8,135},RGB{126,3,168},RGB{204,71,120},RGB{248,149,64},RGB{240,249,33}};
    else if(palette=="cosmic")
        stops={RGB{0,0,0},RGB{20,0,40},RGB{80,0,120},RGB{180,20,200},RGB{255,100,255},RGB{255,220,255}};
    else if(palette=="tropical")
        stops={RGB{0,30,60},RGB{0,120,200},RGB{0,200,150},RGB{100,220,50},RGB{255,200,0},RGB{255,100,0}};
    else // ultra
        stops={RGB{0,0,0},RGB{25,7,107},RGB{9,1,47},RGB{4,4,73},RGB{0,7,100},
               RGB{12,44,138},RGB{24,82,177},RGB{57,125,209},RGB{134,181,229},
               RGB{211,236,248},RGB{241,233,191},RGB{248,201,95},RGB{255,170,0},
               RGB{204,128,0},RGB{153,87,0},RGB{106,52,3},RGB{0,0,0}};

    auto lerp=[&](double t)->RGB{
        if(stops.empty()) return RGB{0,0,0};
        t=std::fmod(t*cycle_speed,1.0); if(t<0)t+=1.0;
        double idx_=t*(stops.size()-1);
        int i0=(int)idx_,i1=std::min(i0+1,(int)stops.size()-1);
        double f=idx_-i0;
        auto& s0=stops[i0]; auto& s1=stops[i1];
        uint8_t r0=std::get<0>(s0),g0=std::get<1>(s0),b0=std::get<2>(s0);
        uint8_t r1=std::get<0>(s1),g1=std::get<1>(s1),b1=std::get<2>(s1);
        return RGB{(uint8_t)(r0+(r1-r0)*f),(uint8_t)(g0+(g1-g0)*f),(uint8_t)(b0+(b1-b0)*f)};
    };

    auto out=py::array_t<uint8_t>({H,W,4});
    auto obuf=out.mutable_unchecked<3>();
    for(int r=0;r<H;++r){
        for(int c=0;c<W;++c){
            double v=rbuf(r,c);
            double t=(vmax>vmin)?(v-vmin)/(vmax-vmin):0.0;
            t=mclamp(t,0.0,1.0);
            if(invert)t=1.0-t;
            t=std::pow(t,gamma);
            auto rgb=lerp(t);
            obuf(r,c,0)=std::get<0>(rgb);obuf(r,c,1)=std::get<1>(rgb);obuf(r,c,2)=std::get<2>(rgb);
            obuf(r,c,3)=255;
        }
    }
    return out;
}

// ═════════════════════════════════════════════════════════════
//  3-D height-map
// ═════════════════════════════════════════════════════════════
py::array_t<float> compute_heightmap(py::array_t<double> raw,double z_scale,bool smooth){
    auto rbuf=raw.unchecked<2>();
    int H=rbuf.shape(0),W=rbuf.shape(1);
    double vmin=1e18,vmax=-1e18;
    for(int r=0;r<H;++r)for(int c=0;c<W;++c){double v=rbuf(r,c);if(v>vmax)vmax=v;if(v<vmin)vmin=v;}
    auto out=py::array_t<float>({H,W});
    auto obuf=out.mutable_unchecked<2>();
    for(int r=0;r<H;++r)for(int c=0;c<W;++c){
        double t=(vmax>vmin)?(rbuf(r,c)-vmin)/(vmax-vmin):0;
        obuf(r,c)=(float)(t*z_scale);
    }
    if(smooth){
        auto tmp=py::array_t<float>({H,W});
        auto tbuf=tmp.mutable_unchecked<2>();
        for(int r=1;r<H-1;++r)for(int c=1;c<W-1;++c){
            float s=0;for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc)s+=obuf(r+dr,c+dc);
            tbuf(r,c)=s/9.0f;
        }
        for(int r=1;r<H-1;++r)for(int c=1;c<W-1;++c)obuf(r,c)=tbuf(r,c);
    }
    return out;
}

// ═════════════════════════════════════════════════════════════
//  IFS (unchanged)
// ═════════════════════════════════════════════════════════════
py::array_t<uint8_t> compute_ifs(
    int width,int height,int num_points,
    const std::vector<std::vector<double>>& transforms,
    const std::string& palette)
{
    auto out=py::array_t<uint8_t>({height,width,4});
    auto obuf=out.mutable_unchecked<3>();
    for(int r=0;r<height;++r)for(int c=0;c<width;++c){obuf(r,c,0)=0;obuf(r,c,1)=0;obuf(r,c,2)=0;obuf(r,c,3)=255;}
    std::vector<double> cdf; double acc=0;
    for(auto&t:transforms){acc+=t[6];cdf.push_back(acc);}
    for(auto&v:cdf)v/=acc;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(0.0,1.0);
    double x=0,y=0,xmin_=-3,xmax_=3,ymin_=-1,ymax_=10;
    {double bx0=1e9,bx1=-1e9,by0=1e9,by1=-1e9,tx=0,ty=0;
     for(int i=0;i<50000;++i){
         double rv=uni(rng);int k=0;
         while(k<(int)cdf.size()-1&&rv>cdf[k])++k;
         auto&t=transforms[k];
         double nx=t[0]*tx+t[1]*ty+t[4],ny=t[2]*tx+t[3]*ty+t[5];
         tx=nx;ty=ny;
         if(i>100){bx0=std::min(bx0,tx);bx1=std::max(bx1,tx);by0=std::min(by0,ty);by1=std::max(by1,ty);}
     }
     xmin_=bx0-0.1;xmax_=bx1+0.1;ymin_=by0-0.1;ymax_=by1+0.1;}
    std::vector<int> density(width*height,0);
    for(int i=0;i<num_points;++i){
        double rv=uni(rng);int k=0;
        while(k<(int)cdf.size()-1&&rv>cdf[k])++k;
        auto&t=transforms[k];
        double nx=t[0]*x+t[1]*y+t[4],ny=t[2]*x+t[3]*y+t[5];
        x=nx;y=ny;
        int px=(int)((x-xmin_)/(xmax_-xmin_)*width);
        int py=(int)((y-ymin_)/(ymax_-ymin_)*height);
        py=height-1-py;
        if(px>=0&&px<width&&py>=0&&py<height)density[py*width+px]++;
    }
    int dmax=*std::max_element(density.begin(),density.end());
    if(dmax==0)dmax=1;
    for(int r=0;r<height;++r)for(int c=0;c<width;++c){
        double t=(double)density[r*width+c]/dmax;t=std::pow(t,0.45);
        uint8_t g1=(uint8_t)(34*t),g2=(uint8_t)(139*t),g3=(uint8_t)(34*t);
        if(palette=="fire"){g1=(uint8_t)(255*t);g2=(uint8_t)(100*t*t);g3=0;}
        else if(palette=="electric"){g1=(uint8_t)(60*t);g2=(uint8_t)(180*t);g3=(uint8_t)(255*t);}
        obuf(r,c,0)=g1;obuf(r,c,1)=g2;obuf(r,c,2)=g3;obuf(r,c,3)=255;
    }
    return out;
}

int get_hardware_threads(){return (int)std::thread::hardware_concurrency();}

// ═════════════════════════════════════════════════════════════
//  pybind11 module
// ═════════════════════════════════════════════════════════════
PYBIND11_MODULE(fractal_core, m){
    m.doc()="Fractal Engine — C++ core via pybind11";

    m.def("compute_mandelbrot",&compute_mandelbrot,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("color_mode")=0,py::arg("bailout")=2.0,
        py::arg("stripe_density")=5.0,py::arg("orbit_trap_x")=0.0,py::arg("orbit_trap_y")=0.0,
        py::arg("num_threads")=4);

    m.def("compute_julia",&compute_julia,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("cr"),py::arg("ci"),
        py::arg("color_mode")=0,py::arg("bailout")=2.0,py::arg("num_threads")=4);

    m.def("compute_burning_ship",&compute_burning_ship,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("bailout")=2.0,py::arg("num_threads")=4);

    m.def("compute_newton",&compute_newton,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("poly_degree")=3,py::arg("tol")=1e-6,py::arg("num_threads")=4);

    m.def("compute_lyapunov",&compute_lyapunov,
        py::arg("amin"),py::arg("amax"),py::arg("bmin"),py::arg("bmax"),
        py::arg("width"),py::arg("height"),
        py::arg("sequence")="AB",py::arg("warmup")=100,py::arg("iters")=200,
        py::arg("num_threads")=4);

    // ── new types ─────────────────────────────────────────────
    m.def("compute_tricorn",&compute_tricorn,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("bailout")=2.0,py::arg("num_threads")=4);

    m.def("compute_multibrot",&compute_multibrot,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("exponent")=3.0,py::arg("bailout")=2.0,py::arg("num_threads")=4);

    m.def("compute_phoenix",&compute_phoenix,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("cr")=0.5667,py::arg("ci")=-0.5,
        py::arg("bailout")=2.0,py::arg("num_threads")=4);

    m.def("compute_burning_ship_julia",&compute_burning_ship_julia,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("cr")=-1.755,py::arg("ci")=0.0,
        py::arg("bailout")=2.0,py::arg("num_threads")=4);

    m.def("compute_nova",&compute_nova,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("poly_degree")=3,py::arg("tol")=1e-6,
        py::arg("bailout")=2.0,py::arg("num_threads")=4);

    m.def("compute_collatz",&compute_collatz,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),py::arg("max_iter"),
        py::arg("bailout")=32.0,py::arg("num_threads")=4);

    m.def("compute_buddhabrot",&compute_buddhabrot,
        py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
        py::arg("width"),py::arg("height"),
        py::arg("max_iter_r")=100,py::arg("max_iter_g")=1000,py::arg("max_iter_b")=5000,
        py::arg("num_samples")=5000000LL,py::arg("num_threads")=4);

    // ── utilities ─────────────────────────────────────────────
    m.def("apply_colormap",&apply_colormap,
        py::arg("raw"),py::arg("palette")="ultra",
        py::arg("gamma")=1.0,py::arg("invert")=false,
        py::arg("smooth_color")=true,py::arg("cycle_speed")=1.0);

    m.def("compute_heightmap",&compute_heightmap,
        py::arg("raw"),py::arg("z_scale")=1.0,py::arg("smooth")=true);

    m.def("compute_ifs",&compute_ifs,
        py::arg("width"),py::arg("height"),py::arg("num_points"),
        py::arg("transforms"),py::arg("palette")="green");

    m.def("get_hardware_threads",&get_hardware_threads);
}
