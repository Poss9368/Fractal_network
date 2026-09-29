"""Barrido reproducible de parámetros FIRE; escribe un informe, no curvas existentes.

python -m src.nlevel.benchmark_fire --sizes 4096 16384 --output /tmp/fire.json
"""
import argparse
import json
from time import perf_counter
import numpy as np
from .dynamic import maximum_length_angle, rest_deformations_from_thetas
from .fire import fire_minimize

PROFILES = {
    'conservative': dict(dt=.1, dt_max=.5, alpha_start=.1),
    'standard': dict(dt=.1, dt_max=1., alpha_start=.1),
    'fast_start': dict(dt=.2, dt_max=1., alpha_start=.1),
    'larger_step': dict(dt=.2, dt_max=1.5, alpha_start=.1),
    'half_step': dict(dt=.5, dt_max=1., alpha_start=.1),
    'unit_step': dict(dt=1., dt_max=1., alpha_start=.1),
    'quick_growth': dict(dt=.2, dt_max=1., alpha_start=.1, finc=1.2),
    'less_mixing': dict(dt=.1, dt_max=1., alpha_start=.05),
    'more_mixing': dict(dt=.2, dt_max=1., alpha_start=.2),
}


def benchmark(sizes, forces, fraction=.5, max_steps=3000, profiles=None):
    # Compilar antes de medir; una sola ejecución secuencial evita que los
    # perfiles compitan entre sí por CPU y memoria.
    warm=np.full(4,.5*maximum_length_angle(4))
    fire_minimize(warm,np.ones(4),np.ones((4,2)),rest_deformations_from_thetas(warm),1.)
    rows=[]
    for name in (PROFILES if profiles is None else profiles):
        params=PROFILES[name]
        for n in sizes:
            rest=np.full(n,fraction*maximum_length_angle(n))
            theta=rest.copy()
            natural=rest_deformations_from_thetas(rest)
            w=np.ones(n); fam=np.ones((n,2))
            for force in forces:
                start=perf_counter()
                result=fire_minimize(theta,w,fam,natural,float(force),**params,
                    max_steps=max_steps,return_info=True,raise_on_failure=False)
                row=dict(profile=name,n=n,force=float(force),seconds=perf_counter()-start,
                    iterations=result.iterations,converged=result.converged,
                    gradient_max=result.gradient_max,energy=result.energy)
                rows.append(row)
                if not result.converged:
                    print('FAILED',name,n,force,result.gradient_max,flush=True)
                    break  # no contaminar la continuación con un estado no convergido
                theta=result.thetas
            group=[r for r in rows if r['profile']==name and r['n']==n]
            print(name,n,'iterations',sum(r['iterations'] for r in group),
                  'seconds',round(sum(r['seconds'] for r in group),3),flush=True)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',nargs='+',type=int,default=[4096,16384])
    parser.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=None)
    parser.add_argument('--points',type=int,default=31)
    parser.add_argument('--fraction',type=float,default=.5)
    parser.add_argument('--max-steps',type=int,default=3000)
    parser.add_argument('--output',default='fire_benchmark.json')
    args=parser.parse_args()
    if any(n < 1 for n in args.sizes) or args.points < 2 or not 0 < args.fraction < 1 or args.max_steps < 1:
        parser.error("Tamaños y pasos positivos, points >= 2, 0 < fraction < 1")
    forces=np.logspace(-5,5,args.points)
    rows=benchmark(args.sizes,forces,args.fraction,args.max_steps,args.profiles)
    with open(args.output,'w') as f:
        json.dump(dict(fraction=args.fraction,ftol=1e-7,profiles=PROFILES,results=rows),f,indent=2)
