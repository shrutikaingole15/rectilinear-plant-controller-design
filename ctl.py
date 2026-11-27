"""Small classical-control toolkit (numpy/scipy only) used to generate every number in the report."""
import numpy as np
from scipy import signal, optimize

P = np.polynomial.polynomial

def pmul(*ps):
    out = np.array([1.0])
    for p in ps:
        out = np.polymul(out, p)
    return out

def padd(a, b):
    return np.polyadd(a, b)

class TF:
    def __init__(self, num, den):
        self.num = np.trim_zeros(np.atleast_1d(np.array(num, float)), 'f')
        self.den = np.trim_zeros(np.atleast_1d(np.array(den, float)), 'f')
        if self.num.size == 0:
            self.num = np.array([0.0])
    def __mul__(self, o):
        if np.isscalar(o):
            return TF(self.num * o, self.den)
        return TF(np.polymul(self.num, o.num), np.polymul(self.den, o.den))
    __rmul__ = __mul__
    def __call__(self, s):
        return np.polyval(self.num, s) / np.polyval(self.den, s)
    def poles(self):
        return np.roots(self.den)
    def zeros(self):
        return np.roots(self.num) if self.num.size > 1 else np.array([])
    def dcgain(self):
        return self(0.0)

def feedback(L):
    """unity negative feedback T = L/(1+L)"""
    return TF(L.num, np.polyadd(L.den, L.num))

def sens(L):
    return TF(L.den, np.polyadd(L.den, L.num))

def freq(L, w):
    return L(1j * w)

def margins(L, wlo=1e-3, whi=1e3, n=200000):
    w = np.logspace(np.log10(wlo), np.log10(whi), n)
    H = freq(L, w)
    mag = np.abs(H)
    ph = np.unwrap(np.angle(H))
    ph_deg = np.degrees(ph)
    # gain crossovers
    idx = np.where(np.diff(np.sign(mag - 1)) != 0)[0]
    wgc_list = []
    for i in idx:
        f = lambda lw: np.abs(L(1j * 10**lw)) - 1
        lw = optimize.brentq(f, np.log10(w[i]), np.log10(w[i + 1]), xtol=1e-14)
        wgc_list.append(10**lw)
    pm_list = []
    for wg in wgc_list:
        k = np.searchsorted(w, wg)
        phg = np.degrees(np.angle(L(1j * wg)))
        # pick branch consistent with unwrapped phase
        ref = ph_deg[min(k, n - 1)]
        phg = phg + 360 * np.round((ref - phg) / 360)
        pm_list.append(180 + phg)
    # phase crossovers (-180 + k*360)
    gm_list = []
    for m in range(-3, 2):
        target = -180 + 360 * m
        idx = np.where(np.diff(np.sign(ph_deg - target)) != 0)[0]
        for i in idx:
            f = lambda lw: np.interp(lw, np.log10(w[i:i+2]), ph_deg[i:i+2]) - target
            # refine with true phase
            def g(lw):
                v = np.degrees(np.angle(L(1j * 10**lw)))
                return ((v - target + 180) % 360) - 180
            try:
                lw = optimize.brentq(g, np.log10(w[i]), np.log10(w[i + 1]), xtol=1e-14)
            except ValueError:
                lw = np.log10(w[i])
            wpc = 10**lw
            gm_list.append((wpc, -20 * np.log10(np.abs(L(1j * wpc)))))
    gm_list.sort()
    return dict(wgc=wgc_list, pm=pm_list, gm=gm_list)

def hinf_S(L, wlo=1e-3, whi=1e3, n=200000):
    w = np.logspace(np.log10(wlo), np.log10(whi), n)
    S = np.abs(1 / (1 + freq(L, w)))
    i = np.argmax(S)
    return S[i], w[i]

def bandwidth(T, wlo=1e-3, whi=1e3, n=200000):
    w = np.logspace(np.log10(wlo), np.log10(whi), n)
    m = 20 * np.log10(np.abs(freq(T, w)))
    m0 = 20 * np.log10(abs(T.dcgain()))
    i = np.where(m < m0 - 3)[0][0]
    f = lambda lw: 20 * np.log10(np.abs(T(1j * 10**lw))) - (m0 - 3)
    return 10**optimize.brentq(f, np.log10(w[i - 1]), np.log10(w[i]))

def step(T, tfinal=10, n=200001, amp=1.0):
    t = np.linspace(0, tfinal, n)
    sys = signal.lti(T.num, T.den)
    t, y = signal.step(sys, T=t)
    return t, amp * y

def stepinfo(t, y, yfinal, band=0.02):
    yss = yfinal
    Mp = max(0.0, (np.max(y) - yss) / yss * 100)
    tp = t[np.argmax(y)]
    a10 = np.where(y >= 0.1 * yss)[0]; a90 = np.where(y >= 0.9 * yss)[0]
    tr = t[a90[0]] - t[a10[0]] if a90.size and a10.size else float('nan')
    out = np.where(np.abs(y - yss) > band * abs(yss))[0]
    ts = t[min(out[-1] + 1, len(t)-1)] if out.size else 0.0
    return dict(Mp=Mp, tp=tp, tr=tr, ts=ts)

def control_effort(C, L, r=0.01, tfinal=10, n=200001):
    """u = C S r  for a step of size r"""
    U = TF(np.polymul(C.num, L.den), np.polymul(C.den, np.polyadd(L.den, L.num)))
    # U may be improper only if C improper; all controllers here are proper
    t = np.linspace(0, tfinal, n)
    sys = signal.lti(U.num, U.den)
    t, u = signal.step(sys, T=t)
    u = r * u
    i = np.argmax(np.abs(u))
    return t, u, abs(u[i]), t[i], np.trapezoid(u**2, t)
