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
    theta_correction: np.ndarray | None = None
    rounded_gradient_max: float = float("nan")
    refinement_iterations: int = 0
    precision: str = "float64"


@njit(cache=True)
def _add_compensated(hi, lo, value):
    total = hi + value
    virtual = total - hi
    error = (hi - (total - virtual)) + (value - virtual)
    tail = lo + error
    result = total + tail
    return result, tail - (result - total)


@njit(cache=True)
def _reference_residual(reference, rest):
    residual = np.empty_like(rest)
    hi, lo = 0., 0.
    for i in range(len(reference)):
        mh, ml = _add_compensated(-hi, -lo, reference[i])
        mh, ml = _add_compensated(mh, ml, -rest[i, 0])
        ph, pl = _add_compensated(hi, lo, reference[i])
        ph, pl = _add_compensated(ph, pl, -rest[i, 1])
        residual[i, 0] = mh + ml
        residual[i, 1] = ph + pl
        hi, lo = _add_compensated(hi, lo, reference[i])
    return residual


@njit(cache=True)
def _product_error(a, b):
    # Producto de dos float64 como parte principal + error (Dekker).
    splitter = 134217729.
    sa, sb = splitter*a, splitter*b
    ah, bh = sa-(sa-a), sb-(sb-b)
    al, bl = a-ah, b-bh
    product = a*b
    error = ((ah*bh-product)+ah*bl+al*bh)+al*bl
    return product, error


@njit(cache=True)
def _length_gradient_small(theta):
    # Forma equivalente a la recurrencia, para cosenos alejados de cero.
    # Compensar sumas y productos evita perder precisión cerca del máximo,
    # donde 1+t_i²-t_i*(1+sum(t)) resulta mucho menor que sus términos.
    tangent = np.tan(theta*.5)
    total, tail = 1., 0.
    logcos, logtail = 0., 0.
    for i in range(len(theta)):
        total, tail = _add_compensated(total, tail, tangent[i])
        quarter = np.sin(theta[i]*.25)
        logcos, logtail = _add_compensated(logcos, logtail, np.log1p(-2*quarter*quarter))
    product = np.exp(logcos+logtail)
    grad = np.empty_like(theta)
    for i in range(len(theta)):
        t = tangent[i]
        tt, tt_error = _product_error(t, t)
        at, at_error = _product_error(t, total)
        hi, lo = _add_compensated(1., 0., tt)
        hi, lo = _add_compensated(hi, lo, -at)
        lo += tt_error-at_error-t*tail
        grad[i] = .5*product*(hi+lo)
    return product*(total+tail), grad


@njit(cache=True)
def _length_gradient(theta):
    if np.max(np.abs(theta)) < np.pi/2:
        return _length_gradient_small(theta)
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
        theta[i] = reference[i] + (z[i] - previous)
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
    # Evaluar el residuo angular antes de restar contribuciones vecinas
    # de carga: reconstruirlo desde esas diferencias amplifica el redondeo.
    suffix, compensation = 0., 0.
    gmax, square = 0., 0.
    for i in range(n-1, -1, -1):
        suffix, compensation = _add_compensated(suffix, compensation, grad[i])
        angular = (suffix - load*gx[i]) + compensation
        gmax = max(gmax, abs(angular))
        square += angular*angular
    for i in range(n-1):
        grad[i] -= load*(gx[i]-gx[i+1])
    grad[-1] -= load*gx[-1]
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
def _angles(reference, z):
    answer = np.empty_like(reference)
    previous = 0.
    for i in range(len(reference)):
        answer[i] = reference[i] + (z[i] - previous)
        previous = z[i]
    return answer


@njit(cache=True)
def _run(reference, residual, rest, weights, families, load, diag, sub,
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
            return _angles(reference, z), energy, gmax, grms, iteration, evaluations, resets, 2
        if gmax <= tolerance or (iteration > 0 and iteration % 200 == 0):
            # Recentrar limita la cancelación al restar sumas acumuladas.
            # La convergencia debe corresponder a los ángulos devueltos,
            # no sólo a una representación interna de sus desplazamientos.
            reference = _angles(reference, z)
            residual = _reference_residual(reference, rest)
            z[:] = 0.
            energy, gs, gmax, grms = _evaluate(z, reference, residual, weights, families, load)
            evaluations += 1
            force = -_lower(diag, sub, gs)
            if np.isfinite(energy) and np.isfinite(grms) and gmax <= tolerance:
                return reference, energy, gmax, grms, iteration, evaluations, resets, 0
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
            return _angles(reference, z), energy, gmax, grms, iteration, evaluations, resets, 3
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
    return _angles(reference, z), energy, gmax, grms, iteration, evaluations, resets, 1


def fire_minimize(thetas, w_i, family_weights, rest_deformations,
                  lambda_restriction, *, ftol=1e-10, rtol=0., max_steps=50000,
                  dt=.1, dt_max=.1, alpha_start=.1, finc=1.1, fdec=.5,
                  falpha=.99, n_min=5, max_step=.05, precondition=True,
                  return_info=False, raise_on_failure=True, theta_correction=None):
    """Alternativa a conjudate_gradient, con los mismos cinco argumentos.

    Para tolerancias < 1e-10, activa refinamiento double-double y requiere
    return_info=True. El estado preciso es el par (result.thetas,
    result.theta_correction); sumarlo en float64 pierde las cifras extra.
    rounded_gradient_max informa el residuo si se descarta la corrección.
    La fase precisa admite |theta| <= 1.5 rad y conserva el mismo objetivo.
    El presupuesto max_steps se comparte entre ambas fases FIRE.

    Devuelve ángulos sin mutar las entradas; return_info=True devuelve
    FIREResult. Por defecto exige max(abs(dH/dtheta)) <= ftol, un criterio
    más fuerte que el RMS del optimizador antiguo. rtol permite añadir
    rtol*max(abs(lambda*dX/dtheta_inicial)) explícitamente.

    Las variables internas son sumas de desplazamientos respecto de la
    entrada, con recentrado periódico y sumas compensadas. La tolerancia
    se comprueba nuevamente sobre los ángulos realmente devueltos. La masa
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
    correction = np.zeros(n) if theta_correction is None else np.array(theta_correction, dtype=float, copy=True)
    if correction.shape != (n,) or not np.all(np.isfinite(correction)):
        raise ValueError("theta_correction debe ser finito y tener forma (N,)")
    # Residuos de los ángulos de entrada con acumulación compensada.
    residual = _reference_residual(theta, rest)
    length, gx = _length_gradient(theta)
    tolerance = ftol+rtol*np.max(np.abs(lambda_restriction*gx))
    precise = tolerance < 1e-10
    if precise and not return_info:
        raise ValueError("ftol < 1e-10 requiere return_info=True para conservar theta_correction")
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
    raw = _run(theta,residual,rest,w,fam,float(lambda_restriction),diagonal,sub,
               max(tolerance,1e-10) if precise else tolerance,
               min(max_steps,1000) if precise else max_steps,
               dt,dt_max,alpha_start,finc,fdec,falpha,n_min,max_step)
    answer,energy,gmax,grms,it,ev,resets,status = raw
    # Informar siempre el residuo del array realmente devuelto, incluso
    # cuando se agota el presupuesto de iteraciones.
    final_residual = _reference_residual(answer, rest)
    energy, _, gmax, grms = _evaluate(np.zeros(n), answer, final_residual, w, fam,
                                      float(lambda_restriction))
    ev += 1
    if status == 0 and (not np.isfinite(grms) or gmax > tolerance):
        status = 4
    refinement_iterations = 0
    rounded_gradient_max = float(gmax)
    precision = "float64"
    if precise and np.isfinite(energy) and np.isfinite(grms):
        try:
            from .fire_precision import refine, evaluate
        except ImportError:
            from fire_precision import refine, evaluate
        if np.max(np.abs(answer)+np.abs(correction)) > 1.5:
            status = 5
        else:
            refined = refine(answer,correction,w,fam,rest,float(lambda_restriction),
                             diagonal,sub,tolerance,max(0,max_steps-it),dt,dt_max,
                             alpha_start,finc,fdec,falpha,n_min,max_step)
            answer,correction,energy,gmax,grms,rit,rev,rresets,status = refined
            it += rit
            ev += rev
            resets += rresets
            refinement_iterations = rit
            precision = "double-double"
            # Comparación explícita: descartar lo puede perder la convergencia.
            rounded_gradient_max = float(evaluate(answer,np.zeros(n),w,fam,rest,
                                                   float(lambda_restriction))[2])
            ev += 1
    else:
        correction = np.zeros(n)
    messages = ('Convergencia alcanzada', 'Límite de iteraciones',
                'Energía o gradiente no finito', 'Paso temporal demasiado pequeño',
                'El residuo final no satisface la tolerancia',
                'El refinamiento preciso requiere |theta| <= 1.5 rad')
    result = FIREResult(answer,status==0,it,ev,float(energy),float(gmax),float(grms),
                        float(tolerance),resets,perf_counter()-start,messages[status],
                        correction,rounded_gradient_max,refinement_iterations,precision)
    if not result.converged and raise_on_failure:
        raise RuntimeError(f'FIRE: {result.message}; N={n}, fuerza={lambda_restriction:.12g}, iter={it}, max|grad|={gmax:.3e}, tolerancia={tolerance:.3e}')
    return result if return_info else answer
