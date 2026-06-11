# kp6_holes.py — Literature benchmark & validation sheet (6B-C)

Scope: anchors for the six-band k.p + Poisson upgrade of the hole-platform g-factor
analysis. Compiled 2026-06-10 from primary sources (arXiv full texts fetched and
quoted verbatim where given). Repo baseline: 4-band hard-wall LK fin model,
gamma1=4.285, gamma2=0.339, gamma3=1.446, kappa=-0.42, giving wire-axis g_x in
[0.4, 1.1] for 7-16 nm fins with the fin axis along [100].

## 1. Six-band valence-band parameter consensus for Si

| Set | g1 | g2 | g3 | kappa | q | Delta_so | Source |
|---|---|---|---|---|---|---|---|
| Niquet-group production set (6-band) | 4.285 | 0.339 | 1.446 | -0.42 | ~0.01 | 44 meV | Venitucci et al., PRB 98, 155319 (2018); arXiv:1807.09185, App. D: "gamma1 = 4.285, gamma2 = 0.339, gamma3 = 1.446 and Delta = 44 meV"; kappa = -0.42 used in their g analysis (Sec. IV C) |
| Cyclotron-resonance origin | 4.28 (A = -4.28 +/- 0.02) | ~0.34 | ~1.45 | -0.42 | ~0.01 (small) | — | Hensel & Feher, Phys. Rev. 129, 1041 (1963), uniaxially stressed Si CR; inverse-mass params convert to the gamma set above |
| Lawaetz computed (five-level k.p) | ~4.22 | ~0.39 | ~1.44 | (not verified) | — | 44 meV | Lawaetz, PRB 4, 3460 (1971). NOTE: original table paywalled; exact kappa attribution unverified. Repo's "Lawaetz 1971" numbers actually match the CR/Winkler set |
| Transport literature (Ottaviani/Reggiani lineage) | 4.22 | 0.39 | 1.44 | — | — | 44 meV | Ottaviani et al. / Reggiani, hole-transport Monte Carlo papers (1970s), values as repeated in e.g. JAP 76, 4192 (1994) |
| Winkler book tables | 4.285 | 0.339 | 1.446 | -0.42 | 0.01 | 44 meV | R. Winkler, *Spin-Orbit Coupling Effects in 2D Electron and Hole Systems* (Springer, 2003), App. tables; the set cited by Voisin 2016 for |g| = 6|kappa| ~ 2.5 |

Recommended set for kp6_holes.py: **gamma1=4.285, gamma2=0.339, gamma3=1.446,
kappa=-0.42, q=0.01, Delta_so=44 meV** — identical to the Venitucci/Niquet 6-band
production codes, giving direct comparability to the best published Si hole-qubit
g calculations. Spread across sets: gamma1 4.22-4.285 (~1.5%), gamma2 0.32-0.39
(~+/-9%, the largest fractional uncertainty; feeds directly into HH-LH splitting),
gamma3 1.44-1.45, Delta_so universally 44 meV.

## 2. Measured Si hole g-tensors in confined geometries

| Ref (DOI/arXiv) | Geometry | Channel axis | g values (axis, field direction) | Conditions |
|---|---|---|---|---|
| Camenzind, Geyer et al., Nat. Electron. 5, 178 (2022); 10.1038/s41928-022-00722-0; arXiv:2103.07369 | Si FinFET, triangular fin, natural Si (100) substrate; gate lengths lB~35 nm, lP~15 nm; effective dot size ~7 nm | fin along **[110]** | "g*-factor of 1.94 +/- 0.05 and 2.35 +/- 0.05" (Q1, Q2), B in-plane **perpendicular to the fin** (i.e. along [-110]) | EDSR slope in f_MW-B plane; 1.5-5 K |
| Geyer et al., Nat. Phys. 20 (2024), 10.1038/s41567-024-02481-5; arXiv:2212.02308 (Supp. S3) | same FinFET platform, triangular fin, two qubits; ~20 nm B/P gates, dot spacing ~40 nm | fin along **[110]** on (100) | Lab frame (x ~ fin [110], y in-plane perp, z [001]): g1 = diag-part (2.31, 2.00, 1.50), g2 = (1.86, 2.76, 1.46); principal g1 = (2.68, 1.68, 1.46), g2 = (3.04, 1.62, 1.42) | full g-tensor from MW spectroscopy at >= 6 field orientations; SO field perp to fin (alpha_so = 93 deg, beta_so = 23 deg), l_so ~ 31 nm |
| Voisin et al., Nano Lett. 16, 88 (2016); 10.1021/acs.nanolett.5b02920; arXiv:1511.08003 | SOI nanowire pMOSFET, channel ~25 nm long, W ~= 10 nm, thickness ~= 10 nm, (100) SOI; first hole | channel along **[110]** | g_par(0h/1h) ~= 2.3 along channel (in-plane values 2.3-2.6 vs azimuth); 1h/2h: in-plane perp component 1.85-2.6 (gate-dependent), along-channel ~2.3 (constant); g_perp([001]) ~= 1.5 (both transitions) | magnetotransport, 260 mK; dg_par/dVg ~= 0.025/mV |
| Crippa et al., PRL 120, 137702 (2018); arXiv:1710.08690 | SOI nanowire FET, 25 nm wide x 8 nm thick, double gate; 10-30 holes/dot | channel along **[110]** | principal |g*| = 2.08, **2.48 (Y ~ nanowire axis)**, 1.62 (Z ~ [001]); single-orientation g_L = 1.96, g_R = 2.02 | 15 mK; G-hat' and g-hat' matrices quoted in paper (Eqs. 5-6) |
| Liles et al., PRB 104, 235303 (2021); arXiv:2012.04985 | planar 28Si MOS dot, 5.9 nm SiO2, single hole, disk ~30 nm diam x ~7 nm thick | planar; x=[110], y=[1-10], z=[001] | g_[1-10] tunable 1.2 +/- 0.1 to 2.6 +/- 0.1 with V_G4; principal (V_G4=-0.9 V): (1.4, 2.3, 3.9), tensor tilted 42 deg off [001]; (V_G4=-0.7 V): (0.3, 1.0, 1.7), tilt 70 deg | dg/dV up to 8.1 +/- 0.2 /V; sweet spot dg/dV=0 observed |

Consistent picture: along/in-plane g for **[110]** channels ~= 1.9-2.6; out-of-plane
g([001]) ~= 1.4-1.6. **Channel-axis flag: every measured device above is
[110]-oriented (or planar with principal axes [110]/[1-10]). The repo model is
[100]-axis aligned; the comparison g_x([100]) vs g_meas([110]) is not
apples-to-apples, and orientation alone is a known O(1) effect (Venitucci &
Niquet, PRB 99, 115317 (2019): [110] dots on (001) "perform much better" and have
qualitatively different g anisotropy than [001]/[100]-oriented dots).**

## 3. Published theory of Si (and Ge-method) hole g beyond 4-band

| Ref | Method | Geometry | Key g results | Split-off? | Electrostatics? |
|---|---|---|---|---|---|
| Venitucci, Bourdet, Pouzada, Niquet, PRB 98, 155319 (2018); arXiv:1807.09185 | **6-band k.p + finite-volume Poisson** (eps_Si=11.7, eps_SiO2=3.9, eps_HfO2=20), Peierls B-coupling | Crippa SOI device, [110] channel, front+back gates | Unstrained HH-like GS: g_z peaks ~5 = 6|kappa| + Delta-g_z; envelope correction Delta-g_z = 2^17 gamma3^2 / (81 pi^4 (3 gamma1 + 10 gamma2)) = **2.14** (their Eq. 25), so g_z ~ 4.66; g_x, g_y rise with HH-LH mixing. Pure-HH refs: g_x=g_y=0, g_z=6|kappa|=2.52; pure-LH: g_x=g_y=4|kappa|=1.68, g_z=0.84 | yes — but "contribution from split-off envelopes is negligible" in GS | yes — essential |
| same, strained | + biaxial strain eps_par | same | eps_par ~ +0.1% flips GS to LH-like; at eps_par=0.2%: **g_x=2.06, g_y=2.41, g_z=0.77** — matches measured anisotropy pattern (g_z smallest) | yes | yes |
| Venitucci & Niquet, PRB 99, 115317 (2019); arXiv:1901.09563 | 6-band-based analytic dot model | box/ellipsoid dots, (001) substrate | [110] vs [001] channel orientation changes g anisotropy and Rabi by O(1); Si best for fast Rabi | yes | model field |
| Bosco, Hetenyi, Loss, PRX Quantum 2, 010348 (2021); arXiv:2011.09417 | **4-band LK** + multipole gate potential (no SO band) | triangular Si fin, L=20 nm, [110]-type axes (DRA) | g_ii(Vg) electrically tunable with sweet spots (their Fig. 8; values figure-only, not digitized here); SOI maximal for triangular cross-section | **no** | yes — inhomogeneous field at fin apex essential |
| Liles et al., PRB 104, 235303 (2021), theory part (Kiselev/Ladd, HRL) | **6x6 LK + Bir-Pikus + gate-stack cool-down strain + self-consistent Schrödinger-Poisson**, full 3D layout | planar Si MOS, single hole | reproduces measured tensor incl. 42-deg tilt only with electrode-induced strain (Delta_HH-LH varies >50% across dot); without BP terms tilt < 1 deg | yes | yes |
| PR Applied 23, 054030 (2025); arXiv:2505.22267 | 6x6 k.p Schrödinger-Poisson + thermal (cool-down) strain | triple-gate triangular Si FinFET (Basel-type) | g-factor and Rabi show "strong strain-dependent variations"; direct sim of Geyer-class device | yes | yes |
| Marcellina et al., PRB 95, 075305 (2017); arXiv:1604.08759 | analytic LK, 2D limit | inversion-asymmetric 2DHGs (Si, Ge, ...) | 2D-limit SOI/Zeeman expressions; useful for the 2D cross-check of kp6 | partial | model field |

2D analytic limits to test against (Winkler 2003): thin (001) Si film, HH ground
state, perturbative Zeeman (no envelope/orbital correction):
|g_z(HH)| = |6 kappa + 13.5 q| = 2.39 (q=0.01) to 2.52 (q=0); |g_par(HH)| ~ 0 (~ q-only);
LH doublet: |g_par| = 4|kappa| = 1.68, |g_z| = 2|kappa| = 0.84.

## 4. What fixes the 4-band hard-wall underprediction of g_x?

Literature verdict, in decreasing order of likely importance for our [0.4, 1.1] vs
1.9-2.4 gap:
1. **Channel orientation.** All benchmark devices are [110]-channel; [100]-aligned
   models give qualitatively different (smaller in-plane) g (Venitucci & Niquet 2019).
   This alone can plausibly account for a large share of the deficit.
2. **Realistic, inhomogeneous electrostatic confinement** (gate-induced field,
   corner/apex localization) enhances HH-LH mixing and raises in-plane g toward and
   beyond the LH value 4|kappa| = 1.68 (Venitucci 2018; Bosco-Hetenyi-Loss 2021 —
   note the latter achieves measured-scale g with only 4 bands once electrostatics
   is realistic).
3. **Orbital (vector-potential/Peierls) terms acting on the envelopes**: the
   B-induced HH-LH envelope mixing adds Delta-g_z = 2.14 in Si beyond the bare
   6|kappa| (Venitucci Eq. 25) — a bare-Zeeman hard-wall diagonalization at B=0
   misses this class of contribution entirely.
4. **Strain** (process- or cool-down-induced, 0.1-0.2% biaxial) reorders HH/LH and
   reproduces the measured anisotropy patterns (Venitucci Fig. 15-16; Liles 2021;
   arXiv:2505.22267).
5. **Split-off coupling per se is NOT the leading fix**: Venitucci 2018 find the
   split-off envelope weight in the ground state "negligible" for their device.
   The 6-band upgrade is still warranted (standard practice of every quantitative
   Si calculation above; Delta_so = 44 meV is comparable to confinement splittings
   at <= 10 nm so 4-band accuracy degrades there), but expectations should be that
   items 1-4, not the SO band, close most of the gap.

## 5. Strain parameters (optional future physics)

Bir-Pikus valence deformation potentials for Si: **b = -2.10 eV, d = -4.85 eV**
(Van de Walle, PRB 39, 1871 (1989) tabulation); measured b = -2.2 +/- 0.2 eV
(Hensel & Feher 1963; d reports spread to ~ -5..-6 eV); a_v ~= 2.46 eV.
Elastic constants c11 = 166 GPa, c12 = 64 GPa (so eps_zz = -0.77 eps_par for
biaxial in-plane strain — Venitucci 2018). Biaxial eps_par ~ +0.1% suffices to
flip HH<->LH ordering in a 25x8 nm [110] channel. Si:B acceptor context:
strain-engineered acceptor spin-orbit qubits, Kobayashi et al., Nat. Mater. 20,
38 (2021). Flagged optional — do not block 6B-D on strain.

## Validation targets for kp6_holes.py

1. Bulk k=0: HH/LH degenerate, SO band split off by exactly 44 meV (tol: machine).
2. Bulk small-k masses along [100]: m_HH = m0/(gamma1 - 2 gamma2) = 0.277 m0,
   m_LH = m0/(gamma1 + 2 gamma2) = 0.202 m0 (tol 1% at k -> 0; LH/SO coupling
   must vanish at k=0 and grow at finite k).
3. 2D HH film limit, (001), bare Zeeman: g_z in [2.39, 2.52] (= |6 kappa + 13.5 q|
   to 6|kappa|), g_in-plane ~ 0 (tol 5%).
4. Box dot with orbital B terms (Lz << Lx, Ly): g_z -> 6|kappa| + 2.14 = 4.66
   (Venitucci Eq. 25; tol 10%; check Lx,Ly-independence of the +2.14).
5. Unstrained Crippa geometry (25 x 8 nm, [110] channel, gate field on): principal
   pattern g_z largest, peak g_z ~= 5, g_x, g_y in 1.5-2.5 (tol +/-20% vs
   Venitucci Fig. 14).
6. Strained Crippa geometry (eps_par = 0.2%): (g_x, g_y, g_z) = (2.06, 2.41, 0.77)
   (tol +/-15%).
7. [110] fin, 7-16 nm, realistic E-field: along-fin g in 1.9-2.5 (Geyer lab-frame
   xx = 2.31/1.86, Camenzind perp-fin 1.94-2.35); g([001]) = 1.42-1.50 (tol +/-0.15).
8. Orientation control: same fin rotated to [100] axes must reproduce the published
   repo regime (g_x ~ 0.4-1.1) — confirming the discrepancy is orientation +
   electrostatics, not a bug.

## Risks / uncertainties

- **Channel-axis mismatch**: benchmark devices are [110]; repo is [100]. Any
  "discrepancy" statement must condition on orientation first.
- **Unknown residual strain** in all devices (0.1-0.2% flips HH/LH); never measured
  independently — strained targets are theory-vs-theory anchors only.
- **Parameter spread**: gamma2 0.32-0.39 (+/-9%) dominates HH-LH splitting
  uncertainty; q rarely measured; Lawaetz original kappa unverified (paywall).
- **Hard-wall vs SiO2 barrier** and image charges; fins localize holes at apex —
  hard-wall-only models miss the dominant confinement scale.
- **Orbital/Peierls B terms required**; bare-Zeeman g misses O(2) contributions.
- **Hole number**: Crippa dots held 10-30 holes (deeper doublet measured);
  Voisin/Liles/Camenzind are single-hole — weight those more.
- Bosco-Hetenyi-Loss g(Vg) values are figure-only (not digitized here); treat as
  qualitative anchor.
- Geyer lab-frame axis assignment (x = fin) inferred from their Fig. 1/3g and the
  SO-field direction (perp to fin); re-verify against published Fig. 3g before
  quoting xx-elements as "along-fin g" in the manuscript.
