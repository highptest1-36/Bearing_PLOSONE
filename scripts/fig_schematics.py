"""Schematics without result values: Fig1 (evaluation design) and Fig2 (architecture).

PLOS ONE: width 7.5 in, Arial 8 pt, flattened RGB TIFF (LZW, 600 dpi). Pseudo-3D blocks are drawn with polygons;
all arrows are horizontal or vertical so that no path crosses a block or a label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8})

OUT = Path(__file__).resolve().parents[1] / "FINAL_PLOS_ONE_PACKAGE" / "figures"
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#3d3c39"
CH, CIR, COR = "#2a78d6", "#eb6834", "#1baf7a"
C_PHYS, C_LAT, C_GATE, C_OUT, C_NEU = "#9cc3f2", "#f5b496", "#8fd9b8", "#c3bdf0", "#e9e8e4"


def _rgb_tif(path):
    """Flattened RGB TIFF, no alpha channel, LZW compression."""
    from PIL import Image
    with Image.open(path) as im:
        rgb = Image.new("RGB", im.size, (255, 255, 255))
        rgb.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        dpi = im.info.get("dpi", (600, 600))
    rgb.save(path, compression="tiff_lzw", dpi=dpi)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    _rgb_tif(OUT / f"{name}.tif")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def _canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _shade(c, f):
    r, g, b = to_rgb(c)
    if f >= 1:
        return tuple(1 - (1 - v) / f for v in (r, g, b))
    return tuple(v * f for v in (r, g, b))


def cuboid(ax, x, y, w, h, d, color, z=2):
    """Pseudo-3D block: front face at (x, y, w, h), depth offset (0.6 d, 0.5 d) towards the upper right."""
    dx, dy = 0.6 * d, 0.5 * d
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True,
                         fc=_shade(color, 1.6), ec=EDGE, lw=0.6, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True,
                         fc=_shade(color, 0.82), ec=EDGE, lw=0.6, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=EDGE, lw=0.6, zorder=z))


def arrow(ax, pts, ls="-", color=INK2):
    """Orthogonal arrow through the given points; the head is on the last segment."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=0.9, ls=ls, zorder=1, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def label(ax, x, y, s, ha="center", va="center", color=INK, **kw):
    ax.text(x, y, s, ha=ha, va=va, color=color, fontsize=8, linespacing=1.15, zorder=6, **kw)


def rbox(ax, x, y, w, h, fc, lw=0.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=EDGE, lw=lw, ls=ls, zorder=2))


def waveform(ax, x, y, w, h, seed, impulsive=True):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 400)
    s = 0.25 * rng.standard_normal(t.size)
    if impulsive:
        for t0 in np.arange(0.04, 1, 0.13):
            k = t >= t0
            s[k] += 1.4 * np.exp(-(t[k] - t0) * 60) * np.sin(2 * np.pi * 90 * (t[k] - t0))
    s = s / np.abs(s).max()
    ax.plot(x + t * w, y + h / 2 + s * h / 2, color=INK2, lw=0.45, zorder=4)


def bearing_icon(ax, x, y, r, color, lw=0.5):
    ax.add_patch(Circle((x, y), r, fc=color, ec=EDGE, lw=lw, zorder=3))
    ax.add_patch(Circle((x, y), 0.55 * r, fc="white", ec=EDGE, lw=lw * 0.8, zorder=4))
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.add_patch(Circle((x + 0.775 * r * np.cos(a), y + 0.775 * r * np.sin(a)), 0.13 * r, fc="white", ec="none", zorder=4))


# ------------------------------------------------------------------ Fig 1: evaluation design
def fig_design():
    W, H = 7.5, 3.55
    fig, ax = _canvas(W, H)
    cls = [CH, CIR, COR]

    # A. development set: 5 rows of one healthy, one IR and one OR bearing
    label(ax, 0.82, 3.38, "PU-BD5 development set", fontweight="bold")
    for j, (c, n) in enumerate(zip(cls, ("H", "IR", "OR"))):
        label(ax, 0.42 + 0.36 * j, 3.12, n)
    for i in range(5):
        label(ax, 0.14, 2.82 - 0.36 * i, f"{i + 1}", color=INK2)
        for j, c in enumerate(cls):
            bearing_icon(ax, 0.42 + 0.36 * j, 2.82 - 0.36 * i, 0.14, c)
    label(ax, 0.82, 0.88, "15 bearings, 64 kHz,\n4 operating settings")

    # B. outer folds: test row dark, training rows light
    label(ax, 3.02, 3.38, "Outer loop: five bearing-disjoint folds", fontweight="bold")
    x0 = 1.78
    for k in range(5):
        xk = x0 + k * 0.52
        label(ax, xk + 0.2, 3.12, f"O{k + 1}", color=INK2)
        for i in range(5):
            for j, c in enumerate(cls):
                test = i == k
                ax.add_patch(Rectangle((xk + 0.13 * j, 2.74 - 0.18 * i), 0.12, 0.15,
                                       fc=c if test else _shade(c, 3.2), ec=EDGE if test else "#b9b8b3", lw=0.4, zorder=3))
    ax.add_patch(Rectangle((1.80, 1.86), 0.13, 0.13, fc=INK2, ec=EDGE, lw=0.4, zorder=3))
    label(ax, 2.0, 1.925, "test row", ha="left")
    ax.add_patch(Rectangle((2.72, 1.86), 0.13, 0.13, fc="#e7e6e2", ec="#b9b8b3", lw=0.4, zorder=3))
    label(ax, 2.92, 1.925, "12 training bearings", ha="left")
    rbox(ax, 1.76, 0.66, 2.56, 0.86, "#f4f3f0")
    label(ax, 3.04, 1.09, "Inner loop: four folds on the 12 training\nbearings of each outer fold. Out-of-fold\n"
          "predictions fit the temperature, the R4\nscore and the architecture-inclusion rule.")
    arrow(ax, [(1.42, 2.10), (1.70, 2.10)])

    # C. final models, frozen
    cuboid(ax, 4.72, 1.80, 0.86, 0.70, 0.30, C_OUT)
    label(ax, 5.15, 2.15, "Final\nmodels")
    label(ax, 5.24, 3.38, "Freeze", fontweight="bold")
    label(ax, 5.24, 1.38, "Trained on all 15\nbearings; models,\nthresholds and\nprotocol hashed")
    arrow(ax, [(4.40, 2.15), (4.66, 2.15)])

    # D. Ottawa, evaluated once
    label(ax, 6.68, 3.38, "Ottawa external set", fontweight="bold")
    for i in range(4):
        for j in range(5):
            b = i * 5 + j
            c = CIR if b < 5 else COR if b < 10 else C_NEU
            bearing_icon(ax, 6.12 + 0.28 * j, 2.92 - 0.30 * i, 0.11, c)
    label(ax, 6.68, 1.62, "20 bearings, 42 kHz;\nhealthy state of all 20,\nIR (1 to 5), OR (6 to 10);\nevaluated once")
    arrow(ax, [(5.86, 2.15), (6.00, 2.15)])

    # legend for bearing colours
    for k, (c, n) in enumerate(((CH, "healthy (H)"), (CIR, "inner race (IR)"), (COR, "outer race (OR)"),
                                (C_NEU, "ball or cage (not in primary task)"))):
        xx = 0.30 + [0, 1.25, 2.62, 4.02][k]
        bearing_icon(ax, xx, 0.20, 0.08, c)
        label(ax, xx + 0.14, 0.20, n, ha="left")
    _save(fig, "Fig1")


# ------------------------------------------------------------------ Fig 2: architecture
def fig_arch():
    W, H = 7.5, 3.85
    fig, ax = _canvas(W, H)
    yt, yb, ym = 2.80, 0.80, 1.80        # centre lines of physics lane, latent lane and gate

    # inputs
    for yc, txt, seed, imp in ((yt, "4-s record, 64 kHz", 1, True), (yb, "1-s windows, 32 kHz", 2, True)):
        rbox(ax, 0.08, yc - 0.30, 0.95, 0.60, "white")
        waveform(ax, 0.14, yc - 0.20, 0.83, 0.40, seed, imp)
        label(ax, 0.555, yc + 0.46, txt)
    label(ax, 0.555, ym + 0.12, "Raw vibration", fontweight="bold")

    # physics lane: envelope spectrum block with spectral lines on its face
    cuboid(ax, 1.30, yt - 0.36, 1.05, 0.72, 0.28, C_PHYS)
    fx = 1.36 + np.linspace(0, 0.93, 60)
    rng = np.random.default_rng(3)
    amp = 0.05 + 0.04 * rng.random(60)
    for k in (9, 18, 27, 36):
        amp[k] = 0.42 if k != 27 else 0.30
    for xx, aa in zip(fx, amp):
        ax.plot([xx, xx], [yt - 0.30, yt - 0.30 + aa], color=INK, lw=0.5, zorder=4)
    label(ax, 1.825, yt + 0.68, "Envelope spectrum", fontweight="bold")
    label(ax, 1.825, yt - 0.52, "fault-frequency regions,\nidentifiability check", va="top")
    arrow(ax, [(1.03, yt), (1.30, yt)])

    # concept vector: 8 stacked cells
    for i in range(8):
        cuboid(ax, 2.78, yt - 0.42 + 0.105 * i, 0.20, 0.105, 0.10, C_PHYS if i % 2 == 0 else "#bcd6f6")
    label(ax, 2.92, yt + 0.68, "Concepts", fontweight="bold")
    label(ax, 2.92, yt - 0.52, "8 concepts\nand masks", va="top")
    arrow(ax, [(2.52, yt), (2.78, yt)])

    # physics head
    cuboid(ax, 3.45, yt - 0.28, 0.62, 0.56, 0.24, C_PHYS)
    label(ax, 3.76, yt, "linear")
    label(ax, 3.83, yt + 0.68, "Physics head", fontweight="bold")
    arrow(ax, [(3.10, yt), (3.45, yt)])

    # latent lane: CNN encoder as shrinking feature maps
    xs = 1.30
    for k, (w, h) in enumerate(((0.12, 0.84), (0.12, 0.72), (0.12, 0.60), (0.12, 0.50), (0.12, 0.40), (0.12, 0.32))):
        cuboid(ax, xs, yb - h / 2, w, h, 0.22, C_LAT)
        xs += 0.21
    label(ax, 1.86, yb + 0.66, "CNN encoder (six blocks)", fontweight="bold")
    arrow(ax, [(1.03, yb), (1.30, yb)])

    cuboid(ax, 3.45, yb - 0.28, 0.62, 0.56, 0.24, C_LAT)
    label(ax, 3.76, yb, "bounded")
    label(ax, 3.83, yb - 0.44, "Latent term", fontweight="bold")
    arrow(ax, [(2.66, yb), (3.45, yb)])

    # gate from label-free evidence quality
    cuboid(ax, 3.45, ym - 0.24, 0.62, 0.48, 0.20, C_GATE)
    label(ax, 3.76, ym, "gate g")
    label(ax, 2.85, ym + 0.10, "evidence\nquality", color=INK2, ha="right")
    arrow(ax, [(2.92, yt - 0.82), (2.92, ym - 0.05), (3.45, ym - 0.05)])

    # fusion node
    fxn = 4.78
    ax.add_patch(Circle((fxn, ym), 0.20, fc="white", ec=EDGE, lw=0.8, zorder=4))
    ax.plot([fxn - 0.11, fxn + 0.11], [ym, ym], color=INK, lw=0.9, zorder=5)
    ax.plot([fxn, fxn], [ym - 0.11, ym + 0.11], color=INK, lw=0.9, zorder=5)
    label(ax, fxn + 0.12, ym + 0.34, "Fusion", fontweight="bold", ha="left")
    arrow(ax, [(4.21, yt), (fxn, yt), (fxn, ym + 0.20)])
    arrow(ax, [(4.21, yb), (fxn, yb), (fxn, ym - 0.20)])
    arrow(ax, [(4.17, ym), (fxn - 0.20, ym)])
    label(ax, 4.40, ym + 0.13, "1 - g", color=INK2)

    # calibrated output: three probability bars
    ox = 5.48
    cuboid(ax, ox, ym - 0.42, 0.80, 0.84, 0.22, C_OUT)
    for k, (c, hgt, n) in enumerate(((CH, 0.14, "H"), (CIR, 0.52, "IR"), (COR, 0.10, "OR"))):
        ax.add_patch(Rectangle((ox + 0.10 + 0.22 * k, ym - 0.24), 0.16, hgt, fc=c, ec=EDGE, lw=0.4, zorder=4))
        label(ax, ox + 0.18 + 0.22 * k, ym - 0.33, n)
    label(ax, ox + 0.47, ym + 0.70, "Calibrated\nprobabilities", fontweight="bold")
    arrow(ax, [(fxn + 0.20, ym), (ox, ym)])

    # verification and selective diagnosis
    for yy, title, txt in ((yt - 0.05, "PEV", "targeted versus\nmatched control"),
                           (yb + 0.05, "PE-SD", "accept or defer\n(R1 to R4)")):
        rbox(ax, 6.56, yy - 0.42, 0.90, 0.84, "#f4f3f0")
        label(ax, 7.01, yy + 0.24, title, fontweight="bold")
        label(ax, 7.01, yy - 0.10, txt)
    arrow(ax, [(6.40, ym + 0.12), (6.46, ym + 0.12), (6.46, yt - 0.05), (6.56, yt - 0.05)])
    arrow(ax, [(6.40, ym - 0.12), (6.46, ym - 0.12), (6.46, yb + 0.05), (6.56, yb + 0.05)])

    # lane captions
    label(ax, 2.65, H - 0.10, "Physics lane (used alone by the physics-only models)", color=INK2, va="top")
    label(ax, 2.65, 0.02, "Latent lane (PAVE-Net only; A6 maps the encoder output to predicted concepts)", color=INK2, va="bottom")
    _save(fig, "Fig2")


if __name__ == "__main__":
    fig_design()
    fig_arch()
    print("ok")
