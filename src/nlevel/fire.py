"""FIRE precondicionado, O(N) en memoria y trabajo por iteración.

Bitzek et al., PRL 97, 170201 (2006), doi:10.1103/PhysRevLett.97.170201.
Se minimiza exactamente E-lambda*X. El precondicionamiento cambia la
métrica de la dinámica ficticia, no la energía ni sus puntos estacionarios.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from numba import njit


@dataclass
class FIREResult:
    thetas: np.ndarray
    converged: bool
    iterations: int
    evaluations: int
    energy: float
    gradient_max: float
    gradient_rms: float
    tolerance: float
    resets: int
    seconds: float
    message: str


@njit(cache=True)
def _length_gradient(theta):
    # Recurrencia y diferenciación inversa: no se divide por cos(theta/2).
    n = len(theta)
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    c = np.cos(theta * .5)
    s = np.sin(theta * .5)
    x[0] = y[0] = 1.
    for i in range(n):
        x[i+1] = c[i]*x[i] + s[i]*y[i]
        y[i+1] = c[i]*y[i]
    grad = np.empty(n)
    ax, ay = 1., 0.
    for i in range(n-1, -1, -1):
        grad[i] = .5*(ax*(-s[i]*x[i]+c[i]*y[i])-ay*s[i]*y[i])
        ay = ax*s[i]+ay*c[i]
        ax *= c[i]
    return x[n], grad


@njit(cache=True)
def _evaluate(z, reference, residual, weights, families, load):
    n = len(z)
    theta = np.empty(n)
    grad = np.zeros(n)
    energy = 0.
    previous = 0.
    for i in range(n):
        theta[i] = reference[i] + z[i] - previous
        rm = residual[i, 0]+z[i]-2*previous
        rp = residual[i, 1]+z[i]
        am = weights[i]*families[i, 0]
        ap = weights[i]*families[i, 1]
        energy += am*rm*rm+ap*rp*rp
        grad[i] += 2*(am*rm+ap*rp)
        if i > 0:
            grad[i-1] -= 4*am*rm
        previous = z[i]
    length, gx = _length_gradient(theta)
    for i in range(n):
        grad[i] -= load*gx[i]
        if i > 0:
            grad[i-1] += load*gx[i]
    # Gradiente respecto de theta, no de las coordenadas transformadas.
    suffix = 0.
    gmax, square = 0., 0.
    for i in range(n-1, -1, -1):
        suffix += grad[i]
        gmax = max(gmax, abs(suffix))
        square += suffix*suffix
    return energy-load*length, grad, gmax, np.sqrt(square/n)


@njit(cache=True)
def _lower(diag, sub, rhs):
    out = np.empty_like(rhs)
    out[0] = rhs[0]/diag[0]
    for i in range(1, len(rhs)):
        out[i] = (rhs[i]-sub[i-1]*out[i-1])/diag[i]
    return out


@njit(cache=True)
def _upper(diag, sub, rhs):
    out = np.empty_like(rhs)
    out[-1] = rhs[-1]/diag[-1]
    for i in range(len(rhs)-2, -1, -1):
        out[i] = (rhs[i]-sub[i]*out[i+1])/diag[i]
    return out


@njit(cache=True)
def _run(reference, residual, weights, families, load, diag, sub,
         tolerance, max_steps, dt, dt_max, alpha_start, finc, fdec,
         falpha, n_min, max_step):
    n = len(reference)
    z = np.zeros(n)
    velocity = np.zeros(n)
    energy, gs, gmax, grms = _evaluate(z, reference, residual, weights, families, load)
    force = -_lower(diag, sub, gs)
    positive, resets, evaluations = 0, 0, 1
    alpha = alpha_start
    iteration = 0
    for iteration in range(max_steps+1):
        if not np.isfinite(energy) or not np.isfinite(gmax) or not np.isfinite(grms):
            return z, energy, gmax, grms, iteration, evaluations, resets, 2
        if gmax <= tolerance:
            return z, energy, gmax, grms, iteration, evaluations, resets, 0
        if iteration == max_steps:
            break
        # Euler semiimplícito para la dinámica FIRE en coordenadas blanqueadas.
        velocity += dt*force
        power = np.dot(velocity, force)
        if power > 0:
            positive += 1
            vn = np.sqrt(np.dot(velocity, velocity))
            fn = np.sqrt(np.dot(force, force))
            if fn > 0:
                velocity = (1-alpha)*velocity+alpha*(vn/fn)*force
            if positive > n_min:
                dt = min(dt*finc, dt_max)
                alpha *= falpha
        else:
            velocity[:] = 0.
            dt *= fdec
            alpha = alpha_start
            positive = 0
            resets += 1
        if dt < 1e-16:
            return z, energy, gmax, grms, iteration, evaluations, resets, 3
        dz = dt*_upper(diag, sub, velocity)
        largest = abs(dz[0])
        for i in range(1,n):
            largest = max(largest, abs(dz[i]-dz[i-1]))
        if largest > max_step:
            factor = max_step/largest
            dz *= factor
            velocity *= factor
        trial = z+dz
        enew, gnew, gmnew, grnew = _evaluate(trial, reference, residual, weights, families, load)
        evaluations += 1
        if not np.isfinite(enew) or not np.isfinite(gmnew) or not np.isfinite(grnew):
            velocity[:] = 0.
            dt *= fdec
            alpha = alpha_start
            positive = 0
            resets += 1
            continue
        z, energy, gs, gmax, grms = trial, enew, gnew, gmnew, grnew
        force = -_lower(diag, sub, gs)
    return z, energy, gmax, grms, iteration, evaluations, resets, 1


def fire_minimize(thetas, w_i, family_weights, rest_deformations,
                  lambda_restriction, *, ftol=1e-7, rtol=0., max_steps=50000,
                  dt=1., dt_max=1., alpha_start=.1, finc=1.1, fdec=.5,
                  falpha=.99, n_min=5, max_step=.05, precondition=True,
                  return_info=False, raise_on_failure=True):
    """Alternativa a conjudate_gradient, con los mismos cinco argumentos.

    Devuelve ángulos sin mutar las entradas; return_info=True devuelve
    FIREResult. Por defecto exige max(abs(dH/dtheta)) <= ftol, un criterio
    más fuerte que el RMS del optimizador antiguo. rtol permite añadir
    rtol*max(abs(lambda*dX/dtheta_inicial)) explícitamente.

    Las variables internas son sumas de desplazamientos respecto de la
    entrada; evitan restar sumas grandes durante cada iteración. La masa
    ficticia tridiagonal aproxima la curvatura elástica y geométrica. No
    se forman matrices densas. precondition=False usa masa unidad en
    esas mismas coordenadas acumuladas (no en theta).

    La fuerza lambda está impuesta; no es una minimización a longitud fija.
    Un fallo de convergencia genera RuntimeError, salvo petición explícita.
    El tiempo reportado incluye compilación JIT si es la primera llamada.
    """
    start = perf_counter()
    theta = np.array(thetas, dtype=float, copy=True, order='C')
    w = np.ascontiguousarray(w_i, dtype=float)
    fam = np.ascontiguousarray(family_weights, dtype=float)
    rest = np.ascontiguousarray(rest_deformations, dtype=float)
    n = theta.size
    if theta.ndim != 1 or n == 0 or w.shape != (n,) or fam.shape != (n,2) or rest.shape != (n,2):
        raise ValueError('Formas requeridas: theta y w_i (N,), familias y reposo (N,2), N>0')
    if any(not np.all(np.isfinite(x)) for x in (theta,w,fam,rest)):
        raise ValueError('Todas las entradas deben ser finitas')
    if np.any(w < 0) or np.any(fam < 0) or np.any(fam > 1):
        raise ValueError('w_i >= 0; pesos de familia en [0,1]')
    if not np.isfinite(lambda_restriction):
        raise ValueError('La fuerza debe ser finita')
    params = (ftol,rtol,dt,dt_max,alpha_start,finc,fdec,falpha,max_step)
    if not all(np.isfinite(x) for x in params):
        raise ValueError('Parámetros FIRE no finitos')
    if ftol <= 0 or rtol < 0 or not 0 < dt <= dt_max or not 0 < alpha_start < 1 or finc <= 1 or not 0 < fdec < 1 or not 0 < falpha <= 1 or max_step <= 0:
        raise ValueError('Parámetros FIRE fuera de rango')
    if isinstance(max_steps,bool) or not isinstance(max_steps,(int,np.integer)) or max_steps < 0 or isinstance(n_min,bool) or not isinstance(n_min,(int,np.integer)) or n_min < 0:
        raise ValueError('max_steps y n_min deben ser enteros no negativos')
    # Misma acumulación inicial que el modelo existente; después sólo se
    # acumulan desplazamientos locales, no ángulos grandes repetidamente.
    prefix = np.r_[0.,np.cumsum(theta[:-1])]
    residual = np.column_stack((theta-prefix-rest[:,0],theta+prefix-rest[:,1]))
    length, gx = _length_gradient(theta)
    tolerance = ftol+rtol*np.max(np.abs(lambda_restriction*gx))
    diagonal = np.ones(n)
    sub = np.zeros(n-1)
    if precondition:
        am, ap = w*fam[:,0], w*fam[:,1]
        curvature = abs(lambda_restriction)*max(abs(length)/4,.25)
        diagonal = 2*(am+ap)+curvature
        diagonal[:-1] += 8*am[1:]+curvature
        off = -4*am[1:]-curvature
        diagonal += max(float(np.max(diagonal)),1.)*1e-12
        # Factorización M=L L^T, con L bidiagonal.
        diagonal[0] = np.sqrt(diagonal[0])
        for i in range(1,n):
            sub[i-1] = off[i-1]/diagonal[i-1]
            diagonal[i] = np.sqrt(max(diagonal[i]-sub[i-1]**2,1e-30))
    raw = _run(theta,residual,w,fam,float(lambda_restriction),diagonal,sub,
               tolerance,max_steps,dt,dt_max,alpha_start,finc,fdec,falpha,n_min,max_step)
    z,energy,gmax,grms,it,ev,resets,status = raw
    answer = theta+np.diff(np.r_[0.,z])
    messages = ('Convergencia alcanzada','Límite de iteraciones','Energía o gradiente no finito','Paso temporal demasiado pequeño')
    result = FIREResult(answer,status==0,it,ev,float(energy),float(gmax),float(grms),
                        float(tolerance),resets,perf_counter()-start,messages[status])
    if not result.converged and raise_on_failure:
        raise RuntimeError(f'FIRE: {result.message}; iter={it}, max|grad|={gmax:.3e}, tolerancia={tolerance:.3e}')
    return result if return_info else answer
