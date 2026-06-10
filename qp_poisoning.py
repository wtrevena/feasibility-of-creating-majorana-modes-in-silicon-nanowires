"""
qp_poisoning.py — quasiparticle-poisoning estimate for the Si:B parent
(pre-submission item 3).  Run: python3 qp_poisoning.py

Standard BCS thermal quasiparticle density fraction:
    x_qp = sqrt(2 pi k_B T_eff / Delta) * exp(-Delta / k_B T_eff)
evaluated at the PARENT spectral gap at the operating field (the reservoir that
poisons the island), for realistic effective QP temperatures T_eff = 50-200 mK
(experiments rarely thermalize QPs below ~100 mK without heroic filtering and
gap engineering). Context thresholds: transmon-community x_qp ~ 1e-9 .. 1e-5;
usable parity lifetimes generally need x_qp <~ 1e-6 with traps/gap engineering.

Operating points (from key_numbers fig8/fig11):
  - all-Si measured-field point:  Si:B at B = 0.33 T -> Delta_p = 91*(1-(B/0.4)^2)
  - all-Si tilted thick parent:   similar scale (~25-30 ueV parent gap)
  - hypothetical Pauli film:      Delta_p = 91 - 57.88*B at B ~ 1.0 -> ~33 ueV
  - Al film reference:            Delta_p ~ 180 ueV at operating field
"""
import json
import numpy as np

KB_UEV_PER_MK = 0.0861733  # ueV per mK

points = {
    "SiB_measured_B0.33T": 91.0 * (1 - (0.33 / 0.4)**2),
    "SiB_pauli_hyp_B1.0T": 91.0 - 57.88 * 1.0,
    "Al_film_B1.0T": 200.0 * (1 - (1.0 / 2.0)**2),
}
Teffs = [50.0, 100.0, 150.0, 200.0]

out = {}
print(f"{'operating point':24s} Dp(ueV) " +
      " ".join(f"x_qp({T:.0f}mK)" for T in Teffs))
for name, Dp in points.items():
    row = {}
    vals = []
    for T in Teffs:
        kT = KB_UEV_PER_MK * T
        x = np.sqrt(2 * np.pi * kT / Dp) * np.exp(-Dp / kT)
        row[f"{T:.0f}mK"] = float(f"{x:.2e}")
        vals.append(f"{x:9.1e}")
    out[name] = dict(parent_gap_ueV=round(Dp, 1), x_qp=row)
    print(f"{name:24s} {Dp:7.1f} " + " ".join(vals))

# headline: temperature at which x_qp = 1e-6 for each point
for name, Dp in points.items():
    Ts = np.linspace(20, 400, 2000)
    x = np.sqrt(2 * np.pi * KB_UEV_PER_MK * Ts / Dp) * \
        np.exp(-Dp / (KB_UEV_PER_MK * Ts))
    T6 = float(Ts[np.searchsorted(x, 1e-6)])
    out[name]["Teff_for_xqp_1e-6_mK"] = round(T6, 0)
    print(f"{name}: T_eff for x_qp=1e-6: {T6:.0f} mK")

kn = json.load(open('output/data/key_numbers.json'))
kn['fig8']['qp_poisoning'] = out
json.dump(kn, open('output/data/key_numbers.json', 'w'), indent=2)
print("written to key_numbers fig8.qp_poisoning")
