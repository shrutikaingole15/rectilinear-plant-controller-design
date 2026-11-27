"""
AE 308/775 Course Project - Group 22 (P2, ECP rectilinear plant, non-collocated x2/v)
Top-level script: regenerates every figure (fig/*.png) and every number (results.json)
quoted in the final report.  Run:  python3 main.py
Requires: numpy, scipy, matplotlib (ctl.py in the same folder).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize
from ctl import TF, feedback, sens, margins, hinf_S, bandwidth, step, stepinfo, control_effort

GROUP = 22
TAG = f"Group {GROUP}"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG, exist_ok=True)
R = {}                       # all reported numbers
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.35,
                     "axes.titlesize": 10, "lines.linewidth": 1.3, "figure.dpi": 150})
BLUE, RED, GRN, ORG, GRY = "#0072BD", "#D95319", "#2E8B57", "#EDB120", "0.45"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)

# ------------------------------------------------------------------ plant data
rho = (0.75, 1.12, 1.00)
m1n, k2n, c1n, c2n = 1.80, 800.0, 25.0, 16.0
m1, k2, c1, c2 = rho[0] * m1n, rho[1] * k2n, rho[2] * c1n, rho[2] * c2n
m2, k1, kf = 1.20, 350.0, 8.0
spec = dict(Mp=16.0, ts=3.32, PM=44.0, GM=8.0, ess=2.0)
r0 = 0.010                                   # 10 mm reference step
Tend = 5.0                                   # length of reference test window [s]

den = [m1 * m2, m1 * c2 + m2 * c1, m1 * k2 + c1 * c2 + m2 * (k1 + k2),
       c1 * k2 + c2 * (k1 + k2), k1 * k2]
G = TF([kf * k2], den)                                   # x2/v  (non-collocated)
G1 = TF(np.array([m2, c2, k2]) * kf, den)                # x1/v  (collocated)
g0 = G.dcgain()
R["den"] = den; R["num"] = [kf * k2]; R["num1"] = list(np.array([m2, c2, k2]) * kf)
R["params"] = dict(m1=m1, m2=m2, k1=k1, k2=k2, c1=c1, c2=c2, kf=kf)
R["poles"] = [[p.real, p.imag] for p in G.poles()]
R["zeros1"] = [[z.real, z.imag] for z in G1.zeros()]
R["dc"] = g0
modes = []
for p in sorted(G.poles(), key=lambda p: abs(p)):
    if p.imag > 0:
        wn = abs(p); modes.append(dict(sigma=-p.real, wd=p.imag, wn=wn, zeta=-p.real / wn))
R["modes"] = modes
z1 = G1.zeros()[0]
R["zero1_wn"] = abs(z1); R["zero1_zeta"] = -z1.real / abs(z1)

wpk = np.logspace(-2, 2, 400001); Gpk = np.abs(G(1j * wpk))
R["G_peak"] = dict(mag=Gpk.max(), dB=20 * np.log10(Gpk.max()), w=wpk[Gpk.argmax()], K_no_crossover=1 / Gpk.max(),
                   dB_at_6=20 * np.log10(abs(G(6j))))
# ------------------------------------------------------------------ Part A figures
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for a, H, name in [(ax[0], G, "G_2(s) = x_2/v"), (ax[1], G1, "G_1(s) = x_1/v")]:
    p = H.poles(); a.plot(p.real, p.imag, "x", ms=8, mew=1.6, color=BLUE, label="poles")
    z = H.zeros()
    if len(z):
        a.plot(z.real, z.imag, "o", ms=7, mfc="none", mew=1.4, color=RED, label="zeros")
    for zeta in [0.2, 0.5, 0.7]:
        th = np.arccos(zeta); rr = np.linspace(0, 45, 2)
        a.plot(-rr * np.cos(th), rr * np.sin(th), ":", color=GRY, lw=0.8)
        a.plot(-rr * np.cos(th), -rr * np.sin(th), ":", color=GRY, lw=0.8)
        a.text(-9.6, 9.6 * np.tan(th) * (1 if 9.6 * np.tan(th) < 44 else 0), "")
    a.set_xlim(-12, 1); a.set_ylim(-45, 45); a.axvline(0, color="k", lw=0.6); a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("Real axis (s$^{-1}$)"); a.set_title(f"{TAG} - Pole-zero map of ${name}$")
    a.legend(loc="lower left", fontsize=7)
ax[0].set_ylabel("Imaginary axis (s$^{-1}$)")
save(fig, "pzmap.png")

t = np.linspace(0, 1.0, 5001)
_, y = step(G, 1.0, 5001)
fig, a = plt.subplots(figsize=(6.0, 2.6))
a.plot(t, 1e3 * y, color=BLUE); a.axhline(1e3 * g0, ls="--", color=GRY, lw=0.9)
a.set_xlabel("Time (s)"); a.set_ylabel("$x_2$ (mm)"); a.set_title(f"{TAG} - Open-loop unit-step response (v = 1 V)")
save(fig, "ol_step.png")
osi = stepinfo(t, y, g0); R["ol_step"] = osi

w = np.logspace(-1, 3, 4000)
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
for H, c, lab in [(G, BLUE, "$G_2$ (non-collocated, $x_2$)"), (G1, RED, "$G_1$ (collocated, $x_1$)")]:
    Hw = H(1j * w)
    ax[0].semilogx(w, 20 * np.log10(abs(Hw)), color=c, label=lab)
    ax[1].semilogx(w, np.degrees(np.unwrap(np.angle(Hw))), color=c)
ax[0].set_ylabel("Magnitude (dB)"); ax[1].set_ylabel("Phase (deg)"); ax[1].set_xlabel("Frequency (rad/s)")
ax[1].set_yticks([-360, -270, -180, -90, 0]); ax[0].legend(fontsize=7)
ax[0].set_title(f"{TAG} - Open-loop Bode plot of the plant")
save(fig, "ol_bode.png")

# ------------------------------------------------------------------ Routh / RL / Nyquist
a4, a3, a2, a1 = den[0], den[1], den[2], den[3]
b1 = (a3 * a2 - a4 * a1) / a3                   # s^2 row first element
# s^1 row: c1 = (b1*a1 - a3*(a0 + 7168K))/b1 = A - B K
A = (b1 * a1 - a3 * den[4]) / b1
B = a3 * kf * k2 / b1
Kcrit = A / B
R["routh"] = dict(b1=b1, A=A, B=B, Kcrit=Kcrit, Kmin=-den[4] / (kf * k2))
cl = np.roots(np.polyadd(den, [Kcrit * kf * k2]))
wcrit = max(abs(cl[np.argmin(abs(cl.real))].imag), 0)
R["routh"]["wcrit"] = wcrit
# auxiliary equation check: b1 s^2 + a0 + 7168 Kcrit = 0
R["routh"]["waux"] = np.sqrt((den[4] + kf * k2 * Kcrit) / b1)

def rlocus(L, Ks):
    return np.array([np.sort_complex(np.roots(np.polyadd(L.den, k * np.pad(L.num, (len(L.den) - len(L.num), 0)))))
                     for k in Ks])

Ks = np.concatenate([np.linspace(0, 400, 4000)])
rl = rlocus(G, Ks)
fig, a = plt.subplots(figsize=(5.6, 4.0))
for i in range(rl.shape[1]):
    a.plot(rl[:, i].real, rl[:, i].imag, ".", ms=1.2, color=BLUE)
p = G.poles(); a.plot(p.real, p.imag, "x", ms=8, mew=1.6, color="k")
a.plot(cl.real[abs(cl.real) < 1e-6], cl.imag[abs(cl.real) < 1e-6], "s", mfc="none", color=RED, ms=7,
       label=f"$K$ = {Kcrit:.4f}, $\\omega$ = {wcrit:.4f} rad/s")
a.axvline(0, color="k", lw=0.6); a.axhline(0, color="k", lw=0.6)
a.set_xlim(-30, 20); a.set_ylim(-60, 60); a.legend(fontsize=7, loc="lower left")
a.set_xlabel("Real axis (s$^{-1}$)"); a.set_ylabel("Imaginary axis (s$^{-1}$)")
a.set_title(f"{TAG} - Root locus of $K\\,G_2(s)$, $0\\leq K\\leq 400$")
save(fig, "rlocus_P.png")

wn_ = np.logspace(-2, 3, 20000)
Hn = Kcrit * G(1j * wn_)
fig, a = plt.subplots(figsize=(5.6, 3.6))
a.plot(Hn.real, Hn.imag, color=BLUE, label="$\\omega>0$"); a.plot(Hn.real, -Hn.imag, "--", color=BLUE, lw=0.9, label="$\\omega<0$")
a.plot(-1, 0, "r+", ms=10, mew=2)
a.set_xlabel("Real axis"); a.set_ylabel("Imaginary axis"); a.legend(fontsize=7)
a.set_title(f"{TAG} - Nyquist plot of $K_{{cr}}G_2(s)$, $K_{{cr}}$ = {Kcrit:.4f}")
a.set_aspect("equal", "datalim")
save(fig, "nyquist_P.png")
# where does G cross the negative real axis
mP = margins(G)
R["G_phase_cross"] = mP["gm"]

# ------------------------------------------------------------------ Part B(1)-(2)
Mp = spec["Mp"] / 100
zeta = -np.log(Mp) / np.sqrt(np.pi**2 + np.log(Mp)**2)
wn = 4 / (zeta * spec["ts"])
sig = 4 / spec["ts"]
wgc_t = wn * np.sqrt(np.sqrt(1 + 4 * zeta**4) - 2 * zeta**2)
PM_t = np.degrees(np.arctan(2 * zeta / np.sqrt(np.sqrt(1 + 4 * zeta**4) - 2 * zeta**2)))
Mr_t = 1 / (2 * zeta * np.sqrt(1 - zeta**2))
wb_t = wn * np.sqrt(1 - 2 * zeta**2 + np.sqrt(4 * zeta**4 - 4 * zeta**2 + 2))
R["spec"] = dict(zeta=zeta, wn=wn, sigma=sig, wd=wn * np.sqrt(1 - zeta**2), wgc=wgc_t, PM=PM_t,
                 theta=np.degrees(np.arccos(zeta)), Mr=Mr_t, wb=wb_t,
                 phG_wgc=np.degrees(np.angle(G(1j * wgc_t))), wgc_max=0.5 * modes[1]["wn"],
                 phG_wmax=np.degrees(np.angle(G(1j * 0.5 * modes[1]["wn"]))),
                 Kp_req=(100 / spec["ess"]) - 1, K_req=((100 / spec["ess"]) - 1) / g0,
                 Kmax_GM=Kcrit / 10**(spec["GM"] / 20))

def region(a, lim=12):
    th = np.arccos(zeta)
    xs = -np.array([sig, lim]);
    a.fill([-sig, -lim, -lim, -sig], [sig * np.tan(th), lim * np.tan(th), -lim * np.tan(th), -sig * np.tan(th)],
           color="#cfe8cf", alpha=0.7, label="admissible region", zorder=0)
    r_ = np.linspace(0, lim / np.cos(th), 2)
    a.plot(-r_ * np.cos(th), r_ * np.sin(th), "--", color=GRN, lw=1); a.plot(-r_ * np.cos(th), -r_ * np.sin(th), "--", color=GRN, lw=1)
    a.axvline(-sig, color=RED, ls="--", lw=1)

fig, a = plt.subplots(figsize=(5.2, 3.8))
region(a, 10)
a.plot([-sig], [sig * np.tan(np.arccos(zeta))], "k^", ms=6); a.plot([-sig], [-sig * np.tan(np.arccos(zeta))], "k^", ms=6)
a.text(-sig + 0.15, sig * np.tan(np.arccos(zeta)) + 0.4, "$s_d$", fontsize=9)
a.text(-9.5, 9.0, f"$\\zeta = {zeta:.3f}$", color=GRN); a.text(-sig + 0.1, -9.3, f"$\\sigma = 4/t_s = {sig:.3f}$", color=RED)
a.set_xlim(-10, 1); a.set_ylim(-10, 10); a.axvline(0, color="k", lw=0.6); a.axhline(0, color="k", lw=0.6)
a.set_xlabel("Real axis (s$^{-1}$)"); a.set_ylabel("Imaginary axis (s$^{-1}$)")
a.set_title(f"{TAG} - Admissible region in the s-plane"); a.legend(fontsize=7, loc="lower left")
save(fig, "region.png")

# ------------------------------------------------------------------ evaluation helper
def evaluate(C, name):
    L = C * G; m = margins(L); T = feedback(L)
    t, y = step(T, Tend, 100001, r0); yf = r0 * T.dcgain()
    si = stepinfo(t, y, yf); siR = stepinfo(t, y, r0)
    Ms, wMs = hinf_S(L)
    tu, u, umax, tumax, E = control_effort(C, L, r0, Tend, 100001)
    uinf = r0 * (C.dcgain() / (1 + L.dcgain())) if np.isfinite(C.dcgain()) else r0 / g0
    Etr = np.trapezoid((u - uinf)**2, tu)
    out = dict(name=name, wgc=m["wgc"], PM=m["pm"], GM=[[a, b] for a, b in m["gm"]], Ms=Ms, wMs=wMs,
               wb=bandwidth(T), Mp=si["Mp"], tp=si["tp"], tr=si["tr"], ts=si["ts"], ts_ref=siR["ts"],
               ess_pct=100 * (r0 - yf) / r0, umax=umax, t_umax=tumax, Eu=E, Eu_tr=Etr, uinf=uinf, u0=u[0],
               clp=[[p.real, p.imag] for p in T.poles()], clz=[[z.real, z.imag] for z in T.zeros()])
    return out, (t, y, tu, u, L, T)

# ------------------------------------------------------------------ Part B(3) baseline P
fK = lambda K: stepinfo(*step(feedback(G * K), 5, 50001), feedback(G * K).dcgain())["Mp"] - spec["Mp"]
K_Mp = optimize.brentq(fK, 10, 45, xtol=1e-6)
R["P"] = {}
Pcases = {"K_Mp": K_Mp, "K_GM": R["spec"]["Kmax_GM"], "K_10": 10.0}
Pdat = {}
for key, K in Pcases.items():
    o, d = evaluate(TF([K], [1]), f"P K={K:.3f}"); o["K"] = K; R["P"][key] = o; Pdat[key] = d
fig, a = plt.subplots(figsize=(6.0, 2.9))
for (key, c) in zip(["K_10", "K_Mp", "K_GM"], [GRY, BLUE, RED]):
    t, y = Pdat[key][0], Pdat[key][1]
    a.plot(t, 1e3 * y, color=c, label=f"K = {Pcases[key]:.2f}  ($e_{{ss}}$ = {R['P'][key]['ess_pct']:.1f}%)")
a.axhline(10, color="k", ls="--", lw=0.9, label="reference (10 mm)")
a.fill_between([0, 1.0], 9.8, 10.2, color="#cfe8cf")
a.set_xlim(0, 1.0); a.set_xlabel("Time (s)"); a.set_ylabel("$x_2$ (mm)"); a.legend(fontsize=7, loc="center right")
a.set_title(f"{TAG} - Baseline proportional control, 10 mm step")
save(fig, "P_step.png")

# ------------------------------------------------------------------ Design 1: PI (Bode construction)
wgc1, PM1 = 3.5, 70.0
phG1 = np.degrees(np.angle(G(1j * wgc1)))
phC1 = -180 + PM1 - phG1
x1 = np.tan(np.radians(phC1 + 90))          # = wgc*Ti
Ti_c = x1 / wgc1
Kp_c = 1 / (np.sqrt(1 + x1**2) / x1 * abs(G(1j * wgc1)))
R["PI_constr"] = dict(wgc=wgc1, PM=PM1, phG=phG1, magG=abs(G(1j * wgc1)), magGdB=20 * np.log10(abs(G(1j * wgc1))),
                      phC=phC1, wTi=x1, Ti=Ti_c, Kp=Kp_c, Ki=Kp_c / Ti_c)
Kp, Ki = 6.0, 150.0                           # rounded final gains
CPI = TF([Kp, Ki], [1, 0])
R["PI"], dPI = evaluate(CPI, "PI"); R["PI"].update(Kp=Kp, Ki=Ki, Ti=Kp / Ki, zero=-Ki / Kp)

# PI alternatives considered (justify the structure: pure I and PID)
CI = TF([Ki], [1, 0]); R["I_only"], _ = evaluate(CI, "I")

# ------------------------------------------------------------------ Design 2: lag compensator
Kc = 50.0
ess_lag = 1.5
Khat = (100 / ess_lag - 1) / g0
beta = Khat / Kc
mKc = margins(G * Kc)
z_first = mKc["wgc"][0] / 10
p_first = z_first / beta
R["LAG_constr"] = dict(Kc=Kc, GM_Kc=mKc["gm"][0][1], wgc_Kc=mKc["wgc"][0], PM_Kc=mKc["pm"][0],
                       Khat=Khat, beta=beta, z_first=z_first, p_first=p_first,
                       ess_Kc=100 / (1 + Kc * g0))
Cfirst = TF([Kc, Kc * z_first], [1, p_first])
R["LAG_first"], dfirst = evaluate(Cfirst, "lag first attempt")
z2, p2 = 4.0, 0.07
CLG = TF([Kc, Kc * z2], [1, p2])
R["LAG"], dLG = evaluate(CLG, "lag"); R["LAG"].update(Kc=Kc, z=z2, p=p2, beta=z2 / p2, Kdc=Kc * z2 / p2)
# slow pole estimate used in the construction
R["LAG_constr"]["slow_pole_est"] = -(p2 + Kc * g0 * z2) / (1 + Kc * g0)
R["LAG_constr"]["lag_phase_at_wgc"] = np.degrees(np.angle((1j * R["LAG"]["wgc"][0] + z2) / (1j * R["LAG"]["wgc"][0] + p2)))

# ------------------------------------------------------------------ design figures
def bode_margin(ax, L, col, lab, w=np.logspace(-2, 3, 6000)):
    Lw = L(1j * w); ax[0].semilogx(w, 20 * np.log10(abs(Lw)), color=col, label=lab)
    ph = np.degrees(np.unwrap(np.angle(Lw)));
    ph = ph - 360 * np.round(ph[0] / 360) if ph[0] > 0 else ph
    ax[1].semilogx(w, ph, color=col)

for key, C, d, col, fname, title in [("PI", CPI, dPI, BLUE, "PI_bode.png", "PI design"),
                                     ("LAG", CLG, dLG, RED, "LAG_bode.png", "lag design")]:
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    bode_margin(ax, G, GRY, "plant $G_2$")
    if key == "LAG":
        bode_margin(ax, Cfirst * G, ORG, f"first attempt ($z_c$ = {z_first:.3f})")
    bode_margin(ax, C * G, col, "$L = C G_2$")
    o = R[key]; wg = o["wgc"][0]; wp, gm = o["GM"][0]
    ax[0].axhline(0, color="k", lw=0.7); ax[1].axhline(-180, color="k", lw=0.7)
    ax[0].plot(wg, 0, "o", color=col); ax[1].plot(wg, -180 + o["PM"][0], "o", color=col)
    ax[1].vlines(wg, -180, -180 + o["PM"][0], color=col, lw=2)
    ax[0].vlines(wp, -gm, 0, color=col, lw=2); ax[1].plot(wp, -180, "s", color=col)
    ax[0].text(wg * 1.15, 6, f"$\\omega_{{gc}}$ = {wg:.3f}", fontsize=7)
    ax[1].text(wg * 1.15, -175 + o["PM"][0], f"PM = {o['PM'][0]:.2f}$^\\circ$", fontsize=7)
    ax[0].text(wp * 1.1, -gm - 12, f"GM = {gm:.2f} dB\n@ {wp:.2f} rad/s", fontsize=7)
    ax[0].set_ylim(-110, 70); ax[1].set_ylim(-370, 10); ax[1].set_yticks([-360, -270, -180, -90, 0])
    ax[0].set_ylabel("Magnitude (dB)"); ax[1].set_ylabel("Phase (deg)"); ax[1].set_xlabel("Frequency (rad/s)")
    ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_title(f"{TAG} - Loop Bode plot, {title}")
    save(fig, fname)

def rl_plot(C, name, fname, Kgrid, Kdes, col):
    L = C * G; Lu = TF(L.num / C.num[0], L.den)          # normalised so that gain = leading controller gain
    rl = rlocus(Lu, Kgrid)
    fig, a = plt.subplots(figsize=(5.6, 3.9))
    region(a, 60)
    for i in range(rl.shape[1]):
        a.plot(rl[:, i].real, rl[:, i].imag, ".", ms=1.0, color=col)
    p = Lu.poles(); a.plot(p.real, p.imag, "x", ms=8, mew=1.5, color="k", label="open-loop poles")
    z = Lu.zeros()
    if len(z): a.plot(z.real, z.imag, "o", mfc="none", ms=7, color="k", label="open-loop zeros")
    cp = np.roots(np.polyadd(Lu.den, Kdes * np.pad(Lu.num, (len(Lu.den) - len(Lu.num), 0))))
    a.plot(cp.real, cp.imag, "s", ms=7, mfc=ORG, mec="k", label=f"closed-loop poles ({name})")
    a.axvline(0, color="k", lw=0.6); a.axhline(0, color="k", lw=0.6)
    a.set_xlim(-30, 2); a.set_ylim(-45, 45); a.legend(fontsize=6.5, loc="lower left")
    a.set_xlabel("Real axis (s$^{-1}$)"); a.set_ylabel("Imaginary axis (s$^{-1}$)")
    a.set_title(f"{TAG} - Root locus, {name}")
    save(fig, fname)

rl_plot(CPI, f"PI, $K_p$ = {Kp:g}", "PI_rlocus.png", np.linspace(0, 40, 6000), Kp, BLUE)
rl_plot(CLG, f"lag, $K_c$ = {Kc:g}", "LAG_rlocus.png", np.linspace(0, 200, 8000), Kc, RED)

# step + effort comparison
fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
for d, col, lab in [(dPI, BLUE, "Design 1: PI"), (dLG, RED, "Design 2: lag")]:
    ax[0].plot(d[0], 1e3 * d[1], color=col, label=lab); ax[1].plot(d[2], d[3], color=col, label=lab)
ax[0].axhline(10 * (1 + spec["Mp"] / 100), color=GRY, ls=":", lw=1, label="$M_p^{max}$ limit")
ax[0].fill_between([0, Tend], 9.8, 10.2, color="#cfe8cf", label="$\\pm$2% band")
ax[0].axvline(spec["ts"], color=GRY, ls="--", lw=1, label="$t_s^{max}$")
ax[0].set_ylabel("$x_2$ (mm)"); ax[0].legend(fontsize=7, loc="lower right", ncol=2); ax[0].set_ylim(0, 12.5)
ax[0].set_title(f"{TAG} - Closed-loop response to the 10 mm reference step")
ax[1].set_ylabel("$u = v$ (V)"); ax[1].set_xlabel("Time (s)"); ax[1].legend(fontsize=7)
ax[1].set_title(f"{TAG} - Control effort")
save(fig, "compare_step_u.png")

# zoom on first 0.6 s of u
fig, a = plt.subplots(figsize=(6.0, 2.5))
for d, col, lab in [(dPI, BLUE, "PI"), (dLG, RED, "lag")]:
    a.plot(d[2], d[3], color=col, label=lab)
a.set_xlim(0, 0.6); a.set_xlabel("Time (s)"); a.set_ylabel("$v$ (V)"); a.legend(fontsize=7)
a.set_title(f"{TAG} - Control effort, first 0.6 s")
save(fig, "u_zoom.png")

# sensitivity
w = np.logspace(-2, 3, 6000)
fig, a = plt.subplots(figsize=(6.0, 3.0))
for C, col, lab in [(CPI, BLUE, "PI"), (CLG, RED, "lag")]:
    L = C * G
    a.semilogx(w, 20 * np.log10(abs(1 / (1 + L(1j * w)))), color=col, label=f"$|S|$ {lab}")
    a.semilogx(w, 20 * np.log10(abs(L(1j * w) / (1 + L(1j * w)))), "--", color=col, label=f"$|T|$ {lab}")
a.axhline(0, color="k", lw=0.6); a.set_ylim(-60, 10)
a.set_xlabel("Frequency (rad/s)"); a.set_ylabel("Magnitude (dB)"); a.legend(fontsize=7, ncol=2)
a.set_title(f"{TAG} - Sensitivity and complementary sensitivity")
save(fig, "sens.png")

# Nyquist of recommended loop (PI)
L = CPI * G
wq = np.logspace(-1.0, 3, 40000)
Lq = L(1j * wq)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
for a, lim in [(ax[0], None), (ax[1], 1.6)]:
    a.plot(Lq.real, Lq.imag, color=BLUE, label="$\\omega>0$"); a.plot(Lq.real, -Lq.imag, "--", color=BLUE, lw=0.8, label="$\\omega<0$")
    a.plot(-1, 0, "r+", ms=10, mew=2)
    th = np.linspace(0, 2 * np.pi, 400); a.plot(np.cos(th), np.sin(th), ":", color=GRY, lw=0.8)
    a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
    a.set_xlabel("Real axis"); a.set_ylabel("Imaginary axis")
if True:
    ax[0].set_xlim(-2, 1); ax[0].set_ylim(-12, 3)
    ax[0].set_title(f"{TAG} - Nyquist of $L$ (PI)")
    ax[1].set_xlim(-1.6, 0.6); ax[1].set_ylim(-1.4, 0.8); ax[1].set_aspect("equal")
    o = R["PI"]; wg = o["wgc"][0]; Lg = L(1j * wg); wp, gm = o["GM"][0]; Lp = L(1j * wp)
    ax[1].plot([0, Lg.real], [0, Lg.imag], color=ORG, lw=1); ax[1].plot(Lg.real, Lg.imag, "o", color=ORG)
    ax[1].plot(Lp.real, Lp.imag, "s", color=GRN)
    ax[1].text(Lg.real + 0.05, Lg.imag - 0.12, f"PM = {o['PM'][0]:.2f}$^\\circ$", fontsize=7, color=ORG)
    ax[1].text(Lp.real - 0.55, 0.25, f"$-1/GM$ = {Lp.real:.4f}", fontsize=7, color=GRN)
    ax[1].set_title(f"{TAG} - zoom near $-1$")
    ax[0].legend(fontsize=7, loc="lower left")
save(fig, "PI_nyquist.png")
R["PI"]["L_at_wpc"] = [Lp.real, Lp.imag]

# closed-loop poles vs region
fig, a = plt.subplots(figsize=(5.6, 3.9))
region(a, 12)
for key, col, mk in [("PI", BLUE, "o"), ("LAG", RED, "s")]:
    cp = np.array([complex(*p) for p in R[key]["clp"]])
    a.plot(cp.real, cp.imag, mk, color=col, mfc="none", ms=8, mew=1.5, label=f"closed-loop poles, {key if key=='PI' else 'lag'}")
a.set_xlim(-12, 1); a.set_ylim(-42, 42); a.axvline(0, color="k", lw=0.6); a.axhline(0, color="k", lw=0.6)
a.set_xlabel("Real axis (s$^{-1}$)"); a.set_ylabel("Imaginary axis (s$^{-1}$)"); a.legend(fontsize=7, loc="lower right")
a.set_title(f"{TAG} - Closed-loop poles and admissible region")
save(fig, "clpoles.png")

# PI: dominant second-order prediction vs actual
cp = [complex(*p) for p in R["PI"]["clp"]]
pair = sorted([p for p in cp if p.imag > 0], key=lambda p: -p.real)[0]
wn_d = abs(pair); z_d = -pair.real / wn_d
R["PI"]["dom"] = dict(sigma=-pair.real, wd=pair.imag, wn=wn_d, zeta=z_d,
                      Mp_pred=100 * np.exp(-np.pi * z_d / np.sqrt(1 - z_d**2)), ts_pred=4 / (-pair.real))
cpL = [complex(*p) for p in R["LAG"]["clp"]]
pairL = sorted([p for p in cpL if p.imag > 0], key=lambda p: -p.real)[0]
R["LAG"]["dom"] = dict(sigma=-pairL.real, wd=pairL.imag, wn=abs(pairL), zeta=-pairL.real / abs(pairL),
                       Mp_pred=100 * np.exp(-np.pi * (-pairL.real / abs(pairL)) / np.sqrt(1 - (pairL.real / abs(pairL))**2)),
                       ts_pred=4 / (-pairL.real))
# zeta of all PI / lag complex poles
for key in ["PI", "LAG"]:
    R[key]["clp_zeta"] = [[p[0], p[1], -p[0] / np.hypot(*p)] for p in R[key]["clp"]]

# second-order reference vs actual PI response
t = dPI[0]
s2 = TF([wn_d**2], [1, 2 * z_d * wn_d, wn_d**2]); _, y2 = step(s2, Tend, 100001, r0)
real_p = [p for p in cp if abs(p.imag) < 1e-9][0]
s3 = TF([wn_d**2 * (-real_p.real)], np.polymul([1, 2 * z_d * wn_d, wn_d**2], [1, -real_p.real])); _, y3 = step(s3, Tend, 100001, r0)
fig, a = plt.subplots(figsize=(6.0, 2.8))
a.plot(t, 1e3 * dPI[1], color=BLUE, label="full closed loop (PI)")
a.plot(t, 1e3 * y2, "--", color=ORG, label=f"2nd-order approx. ($\\zeta$={z_d:.3f}, $\\omega_n$={wn_d:.3f})")
a.plot(t, 1e3 * y3, ":", color=GRN, lw=1.6, label=f"2nd-order + real pole at {real_p.real:.3f}")
a.set_xlim(0, 1.5); a.set_xlabel("Time (s)"); a.set_ylabel("$x_2$ (mm)"); a.legend(fontsize=7, loc="lower right")
a.set_title(f"{TAG} - Dominant-pole prediction vs simulation (PI)")
save(fig, "PI_dominant.png")
R["PI"]["approx3"] = stepinfo(t, y3, r0); R["PI"]["approx2"] = stepinfo(t, y2, r0)

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, complex): return [o.real, o.imag]
    return o
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
    json.dump(clean(R), f, indent=1)
print("done")

# ------------------------------------------------------------------ residues of the step response (Part C(4))
from scipy.signal import residue
for key, C in [("PI", CPI), ("LAG", CLG)]:
    T = feedback(C * G)
    rr, pp, _ = residue(r0 * T.num, np.polymul(T.den, [1, 0]))
    R[key]["residues"] = [[p.real, p.imag, abs(r) * 1e3] for r, p in zip(rr, pp)]   # |residue| in mm
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
    json.dump(clean(R), f, indent=1)
