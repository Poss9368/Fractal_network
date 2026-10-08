"""Refinamiento FIRE con ángulos hi+lo y aritmética double-double.

No modifica el objetivo ni usa tolerancias relativas. Los pares hi+lo
se conservan al devolver el estado: redondearlos a un solo float64 puede
volver a aumentar el residuo. Dominio trigonométrico: |theta| <= 1.5 rad.
"""
import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def add(ah, al, bh, bl):
    s = ah+bh
    v = s-ah
    e = (ah-(s-v))+(bh-v)+al+bl
    h = s+e
    return h, e-(h-s)


@njit(cache=True, inline='always')
def mul(ah, al, bh, bl):
    splitter = 134217729.
    sa, sb = splitter*ah, splitter*bh
    a1, b1 = sa-(sa-ah), sb-(sb-bh)
    a2, b2 = ah-a1, bh-b1
    p = ah*bh
    e = ((a1*b1-p)+a1*b2+a2*b1)+a2*b2
    e += ah*bl+al*bh+al*bl
    h = p+e
    return h, e-(h-p)


@njit(cache=True, inline='always')
def div(ah, al, bh, bl):
    q = ah/bh
    ph, pl = mul(bh, bl, q, 0.)
    rh, rl = add(ah, al, -ph, -pl)
    return add(q, 0., (rh+rl)/bh, 0.)


@njit(cache=True)
def sincos(hi, lo):
    # 16 términos: error de truncamiento < 4e-40 para |hi+lo| <= .75.
    xxh, xxl = mul(hi, lo, hi, lo)
    sh, sl, ch, cl = hi, lo, 1., 0.
    sth, stl, cth, ctl = hi, lo, 1., 0.
    for k in range(1, 16):
        sth, stl = mul(sth, stl, -xxh, -xxl)
        sth, stl = div(sth, stl, float(2*k*(2*k+1)), 0.)
        cth, ctl = mul(cth, ctl, -xxh, -xxl)
        cth, ctl = div(cth, ctl, float((2*k-1)*(2*k)), 0.)
        sh, sl = add(sh, sl, sth, stl)
        ch, cl = add(ch, cl, cth, ctl)
    return sh, sl, ch, cl


@njit(cache=True)
def angles(reference, correction, z):
    hi, lo = np.empty_like(z), np.empty_like(z)
    previous = 0.
    for i in range(len(z)):
        dh, dl = add(z[i], 0., -previous, 0.)
        hi[i], lo[i] = add(reference[i], correction[i], dh, dl)
        previous = z[i]
    return hi, lo


@njit(cache=True)
def evaluate(hi, lo, weights, families, rest, load):
    n = len(hi)
    tangent_h, tangent_l = np.empty(n), np.empty(n)
    direct_h, direct_l = np.empty(n), np.empty(n)
    future_h, future_l = np.empty(n), np.empty(n)
    ph, pl, total_h, total_l = 1., 0., 1., 0.
    prefix_h, prefix_l, eh, el = 0., 0., 0., 0.
    for i in range(n):
        if abs(hi[i]+lo[i]) > 1.5 or not np.isfinite(hi[i]+lo[i]):
            return np.nan, np.full(n, np.nan), np.inf, np.inf
        sh, sl, ch, cl = sincos(hi[i]*.5, lo[i]*.5)
        th, tl = div(sh, sl, ch, cl)
        tangent_h[i], tangent_l[i] = th, tl
        ph, pl = mul(ph, pl, ch, cl)
        total_h, total_l = add(total_h, total_l, th, tl)
        rmh, rml = add(hi[i], lo[i], -prefix_h, -prefix_l)
        rmh, rml = add(rmh, rml, -rest[i,0], 0.)
        rph, rpl = add(hi[i], lo[i], prefix_h, prefix_l)
        rph, rpl = add(rph, rpl, -rest[i,1], 0.)
        amh, aml = mul(weights[i], 0., families[i,0], 0.)
        aph, apl = mul(weights[i], 0., families[i,1], 0.)
        mh, ml = mul(amh, aml, rmh, rml)
        pwh, pwl = mul(aph, apl, rph, rpl)
        dh, dl = add(mh, ml, pwh, pwl)
        fh, fl = add(-mh, -ml, pwh, pwl)
        direct_h[i], direct_l[i] = 2*dh, 2*dl
        future_h[i], future_l[i] = 2*fh, 2*fl
        emh, eml = mul(mh, ml, rmh, rml)
        eph, epl = mul(pwh, pwl, rph, rpl)
        eh, el = add(eh, el, emh, eml)
        eh, el = add(eh, el, eph, epl)
        prefix_h, prefix_l = add(prefix_h, prefix_l, hi[i], lo[i])
    xh, xl = mul(ph, pl, total_h, total_l)
    fxh, fxl = mul(load, 0., xh, xl)
    eh, el = add(eh, el, -fxh, -fxl)
    gh, gl = np.empty(n), np.empty(n)
    suffix_h, suffix_l, gmax, square = 0., 0., 0., 0.
    for i in range(n-1, -1, -1):
        th, tl = tangent_h[i], tangent_l[i]
        tth, ttl = mul(th, tl, th, tl)
        ath, atl = mul(th, tl, total_h, total_l)
        kh, kl = add(1., 0., tth, ttl)
        kh, kl = add(kh, kl, -ath, -atl)
        xgh, xgl = mul(ph*.5, pl*.5, kh, kl)
        fgh, fgl = mul(load, 0., xgh, xgl)
        dh, dl = add(direct_h[i], direct_l[i], suffix_h, suffix_l)
        gh[i], gl[i] = add(dh, dl, -fgh, -fgl)
        value = gh[i]+gl[i]
        gmax = max(gmax, abs(value))
        square += value*value
        suffix_h, suffix_l = add(suffix_h, suffix_l, future_h[i], future_l[i])
    gradient = np.empty(n)
    for i in range(n-1):
        dh, dl = add(gh[i], gl[i], -gh[i+1], -gl[i+1])
        gradient[i] = dh+dl
    gradient[-1] = gh[-1]+gl[-1]
    return eh+el, gradient, gmax, np.sqrt(square/n)


@njit(cache=True)
def lower(diag, sub, rhs):
    out = np.empty_like(rhs)
    out[0] = rhs[0]/diag[0]
    for i in range(1, len(rhs)):
        out[i] = (rhs[i]-sub[i-1]*out[i-1])/diag[i]
    return out


@njit(cache=True)
def upper(diag, sub, rhs):
    out = np.empty_like(rhs)
    out[-1] = rhs[-1]/diag[-1]
    for i in range(len(rhs)-2, -1, -1):
        out[i] = (rhs[i]-sub[i]*out[i+1])/diag[i]
    return out


@njit(cache=True)
def refine(reference, correction, weights, families, rest, load, diag, sub,
           tolerance, max_steps, dt, dt_max, alpha_start, finc, fdec,
           falpha, n_min, max_step):
    z, velocity = np.zeros(len(reference)), np.zeros(len(reference))
    energy, gs, gmax, grms = evaluate(reference, correction, weights, families, rest, load)
    force = -lower(diag, sub, gs)
    alpha, positive, resets, evaluations = alpha_start, 0, 0, 1
    hi, lo = reference.copy(), correction.copy()
    for iteration in range(max_steps+1):
        if not np.isfinite(energy) or not np.isfinite(grms):
            return hi, lo, energy, gmax, grms, iteration, evaluations, resets, 2
        if gmax <= tolerance:
            return hi, lo, energy, gmax, grms, iteration, evaluations, resets, 0
        if iteration == max_steps:
            break
        velocity += dt*force
        if np.dot(velocity, force) > 0:
            positive += 1
            fn, vn = np.linalg.norm(force), np.linalg.norm(velocity)
            if fn > 0:
                velocity = (1-alpha)*velocity+alpha*(vn/fn)*force
            if positive > n_min:
                dt = min(dt*finc, dt_max)
                alpha *= falpha
        else:
            velocity[:] = 0.
            dt *= fdec
            alpha, positive = alpha_start, 0
            resets += 1
        if dt < 1e-16:
            return hi, lo, energy, gmax, grms, iteration, evaluations, resets, 3
        dz = dt*upper(diag, sub, velocity)
        largest = abs(dz[0])
        for i in range(1,len(dz)):
            largest = max(largest, abs(dz[i]-dz[i-1]))
        if largest > max_step:
            dz *= max_step/largest
            velocity *= max_step/largest
        trial = z+dz
        th, tl = angles(reference, correction, trial)
        enew, gnew, gmnew, grnew = evaluate(th, tl, weights, families, rest, load)
        evaluations += 1
        if not np.isfinite(enew) or not np.isfinite(grnew):
            velocity[:] = 0.
            dt *= fdec
            alpha, positive = alpha_start, 0
            resets += 1
            continue
        z, hi, lo = trial, th, tl
        energy, gs, gmax, grms = enew, gnew, gmnew, grnew
        force = -lower(diag, sub, gs)
    return hi, lo, energy, gmax, grms, iteration, evaluations, resets, 1
