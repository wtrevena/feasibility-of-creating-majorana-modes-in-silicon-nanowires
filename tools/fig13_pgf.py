"""Render fig13 (convergence summary) as vector graphics via pgfplots,
directly from output/data/convergence.json — a network/matplotlib-free
generator producing the same four panels as convergence.make_figure.
Output: output/fig13_convergence.pdf (+ .png preview via pdftoppm).
Usage: python tools/fig13_pgf.py   (requires pdflatex + pgfplots)
"""
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
res = json.load(open(os.path.join(OUT, "data", "convergence.json")))

A = res["A_dx"]; B = res["B_L"]; C = res["C_seeds"]; G = res["G_grid"]

def coords(pairs):
    return " ".join(f"({x},{y})" for x, y in pairs)

dxs = [10, 5, 2.5]
a_clean = coords((d, A[f"dx={d:g}nm"]["clean"]) for d in dxs)
a_step = coords((d, A[f"dx={d:g}nm"]["step085"]) for d in dxs)

Ls = [1.25, 2.5, 5.0]
b_clean = coords((L, B[f"L={L}um"]["clean"]) for L in Ls)
b_step = coords((L, B[f"L={L}um"]["step085"]) for L in Ls)
b_ens = coords((L, B[f"L={L}um"]["ens50_median"]) for L in Ls)

ns = sorted(int(k) for k in C["running_median"])
c_med = coords((n, C["running_median"][str(n)]) for n in ns)
lo, hi = C["bootstrap95"]

glabels = list(next(iter(G.values())).keys())
gx = list(range(len(glabels)))
gseries = "\n".join(
    r"\addplot+[mark=*, mark size=1.5pt] coordinates {"
    + coords((i, G[key][l]["gap"]) for i, l in enumerate(glabels))
    + "};\n" + r"\addlegendentry{" + key.replace("_", r"\_") + "}"
    for key in G)

tex = r"""\documentclass[tikz]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17,
  every axis/.append style={font=\scriptsize, grid=major,
    grid style={gray!25}, legend style={font=\tiny, draw=gray!50},
    label style={font=\scriptsize}, title style={font=\scriptsize}},
  width=7.2cm, height=5.4cm}
\begin{document}
\begin{tikzpicture}
\begin{axis}[name=a, title={(a) lattice-spacing convergence ($L=2.5\,\mu$m)},
  xlabel={$dx$ (nm)}, ylabel={$E_2$ ($\mu$eV)}, x dir=reverse,
  ymin=0, ymax=25, legend pos=south west]
\addplot+[mark=*] coordinates {%(a_clean)s};\addlegendentry{clean wedge gap}
\addplot+[mark=square*] coordinates {%(a_step)s};
\addlegendentry{single step $\Delta\varphi=0.85\pi$}
\draw[gray, dotted, thick] (axis cs:5,0) -- (axis cs:5,25)
  node[pos=0.55, right, font=\tiny, gray]{production};
\end{axis}
\begin{axis}[name=b, at={(a.right of north east)}, anchor=left of north west,
  xshift=8mm, title={(b) length convergence ($dx=5$\,nm)},
  xlabel={$L$ ($\mu$m)}, ylabel={$E_2$ ($\mu$eV)},
  ymin=0, ymax=25, legend pos=north east]
\addplot+[mark=*] coordinates {%(b_clean)s};\addlegendentry{clean}
\addplot+[mark=square*] coordinates {%(b_step)s};\addlegendentry{single step}
\addplot+[mark=diamond*] coordinates {%(b_ens)s};
\addlegendentry{50-nm ensemble median$^{\dagger}$}
\draw[gray, dotted, thick] (axis cs:2.5,0) -- (axis cs:2.5,25)
  node[pos=0.55, right, font=\tiny, gray]{production};
\end{axis}
\begin{axis}[name=c, at={(a.below south west)}, anchor=above north west,
  yshift=-8mm, title={(c) seed convergence, 50-nm same-sign ensemble},
  xlabel={number of disorder seeds}, ylabel={median $E_2$ ($\mu$eV)},
  ymin=0.4, ymax=0.9, legend pos=north east]
\addplot[fill=blue!12, draw=none] coordinates
  {(%(n0)d,%(lo)s) (%(n1)d,%(lo)s) (%(n1)d,%(hi)s) (%(n0)d,%(hi)s)}
  -- cycle;
\addlegendentry{56-seed bootstrap 95\%% CI}
\addplot+[mark=*, blue] coordinates {%(c_med)s};
\addlegendentry{running median}
\end{axis}
\begin{axis}[name=d, at={(c.right of north east)}, anchor=left of north west,
  xshift=8mm, title={(d) hole-platform optimizer-grid convergence},
  xlabel={grid $n_B{\times}n_\Delta{\times}n_\mu$},
  ylabel={best gap ($\mu$eV)}, ymin=0, ymax=35,
  xtick={%(gx)s}, xticklabels={%(glabels)s},
  x tick label style={font=\tiny}, legend pos=outer north east,
  legend style={font=\tiny}]
%(gseries)s
\end{axis}
\end{tikzpicture}
\end{document}
""" % dict(a_clean=a_clean, a_step=a_step, b_clean=b_clean, b_step=b_step,
           b_ens=b_ens, c_med=c_med, lo=lo, hi=hi, n0=ns[0], n1=ns[-1],
           gx=",".join(map(str, gx)),
           glabels=",".join("{" + l + "}" for l in glabels),
           gseries=gseries)

with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "fig13.tex")
    open(src, "w").write(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode",
                    "-output-directory", td, src],
                   check=True, capture_output=True)
    pdf = os.path.join(td, "fig13.pdf")
    dst = os.path.join(OUT, "fig13_convergence.pdf")
    open(dst, "wb").write(open(pdf, "rb").read())
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", dst,
                    os.path.join(OUT, "fig13_convergence")], check=True)
print("wrote output/fig13_convergence.pdf and .png")
