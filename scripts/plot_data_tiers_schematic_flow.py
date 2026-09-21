#!/usr/bin/env python3
"""
Data Tiers Schematic, flow version (L0 -> L1 -> L2)
====================================================
Variant of `scripts/plot_data_tiers_schematic.py`: outline-only tier boxes and
Sankey-style arrows whose thickness is proportional to row count, so the share
of rows carried forward (solid arrow) and dropped (hollow arrow) at each step
can be read off the figure. One scale is shared across both steps, so the two
steps are directly comparable with each other and with the box row counts.

Box outlines and arrows share a single accent color (ACCENT). Counts are
hardcoded below (same convention as the pasted script this was based on);
replace them with values from the dataset build when the dataset changes.

Output:
  figs/plot_data_tiers_schematic_flow/<timestamp>/data_tiers_schematic.{png,svg}

Usage:
    python scripts/plot_data_tiers_schematic_flow.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.log_paths import timestamp as _run_timestamp, update_latest  # noqa: E402

# Match the paper's Computer Modern look via Latin Modern
for pattern in (
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf",
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
):
    for f in glob.glob(pattern):
        fm.fontManager.addfont(f)
mpl.rcParams.update({
    "font.family": "Latin Modern Roman",
    "mathtext.fontset": "cm",
    "svg.fonttype": "path",
    "font.size": 9,
})

# ---- data (replace with values computed from the dataset build) ----
tiers = [
    dict(name="L0", desc="Every whole second at which\nany instrument in a campaign\nreported a measurement",
         rows=4_572_581, cols=46, camps=15, col_note=None),
    dict(name="L1", desc="One row per CPI particle\nimage, joined to its\nexact-second L0 record",
         rows=2_997_447, cols=47, camps=12, col_note="+1"),
    dict(name="L2", desc="L1 rows with all seven\ncore variables present:\n$T,\\ p,\\ S_i,\\ q_v$, lat, lon, alt",
         rows=1_828_818, cols=47, camps=11, col_note=None),
]
ops = [
    dict(label="exact-second join\nwith CPI image\ntimestamps",
         drop="−3 campaigns\nESCAPE, OLYMPEX, POSIDON\n(no CPI imagery)"),
    dict(label="core-variable\ncompleteness\nfilter",
         drop="−1 campaign\nMPACE\n(no water-vapor instrument)"),
]

# One accent color for box outlines AND arrows (change here to recolor both)
ACCENT = "#1f5a8f"
ink = "#1a1a1a"
muted = "#4d4d4d"

W, H = 7.0, 2.5
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.08, W + 0.08); ax.set_ylim(0.0, H); ax.axis("off")

bw, bh, gap = 1.78, 1.72, 0.86
x0 = (W - (3 * bw + 2 * gap)) / 2
y0 = 0.72
hh = 0.36  # header height

# Arrow thickness (inches) per row: the full L0 count maps to MAX_BAND.
MAX_BAND = 0.56
K = MAX_BAND / tiers[0]["rows"]


def fmt_rows(n):
    return f"{n/1e6:.2f} M rows"


def curve(p0, p3, frac=0.6):
    """Cubic bezier leaving p0 horizontally and arriving at p3 vertically."""
    (xa, ya), (xb, yb) = p0, p3
    return [(xa + frac * (xb - xa), ya), (xb, ya + frac * (yb - ya)), (xb, yb)]


def drop_band(xa, y_bot, d, xm, y_end, head_len=0.13, head_extra=0.05):
    """Band of thickness d that leaves horizontally at xa (bottom edge y_bot)
    and turns downward, ending in an arrowhead at y_end, centred on xm."""
    y_top = y_bot + d
    y_turn = y_bot - 0.20 - d            # where the band becomes vertical
    xr, xl = xm + d / 2, xm - d / 2
    y_head = y_end + head_len
    verts, codes = [], []

    def add(pt, code):
        verts.append(pt); codes.append(code)

    add((xa, y_top), MPath.MOVETO)                    # top edge -> right side
    for c in curve((xa, y_top), (xr, y_turn)):
        add(c, MPath.CURVE4)
    add((xr, y_head), MPath.LINETO)
    add((xr + head_extra, y_head), MPath.LINETO)      # arrowhead
    add((xm, y_end), MPath.LINETO)
    add((xl - head_extra, y_head), MPath.LINETO)
    add((xl, y_head), MPath.LINETO)
    add((xl, y_turn), MPath.LINETO)                   # up the left side
    add((xl, y_turn + 0.6 * (y_bot - y_turn)), MPath.CURVE4)   # bottom edge, back to xa
    add((xa + 0.6 * (xl - xa), y_bot), MPath.CURVE4)
    add((xa, y_bot), MPath.CURVE4)
    add((xa, y_top), MPath.CLOSEPOLY)
    return MPath(verts, codes)


for i, t in enumerate(tiers):
    x = x0 + i * (bw + gap)
    # outline-only box; interior is transparent
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="none", ec=ACCENT, lw=1.1, zorder=3))
    cx = x + bw / 2
    ax.text(cx, y0 + bh - hh / 2, t["name"], ha="center", va="center",
            fontsize=12, fontweight="bold", color=ink, zorder=4)
    ax.plot([x, x + bw], [y0 + bh - hh, y0 + bh - hh], color=ACCENT, lw=0.8, zorder=4)
    ax.text(cx, y0 + bh - hh - 0.12, t["desc"], ha="center", va="top",
            fontsize=8, color=ink, linespacing=1.3, zorder=4)
    ax.plot([x + 0.18, x + bw - 0.18], [y0 + 0.66, y0 + 0.66], color=ACCENT, lw=0.5, alpha=0.6, zorder=4)
    ax.text(cx, y0 + 0.43, fmt_rows(t["rows"]), ha="center", va="center",
            fontsize=11, fontweight="bold", color=ink, zorder=4)
    cols = f"{t['cols']} columns" + (f" ({t['col_note']})" if t["col_note"] else "")
    ax.text(cx, y0 + 0.17, f"{cols} · {t['camps']} campaigns", ha="center", va="center",
            fontsize=7.6, color=muted, zorder=4)

# flow arrows: solid = rows carried forward, hollow = rows dropped
for i, op in enumerate(ops):
    n_in, n_out = tiers[i]["rows"], tiers[i + 1]["rows"]
    s, r = n_in * K, n_out * K
    d = s - r
    xa = x0 + (i + 1) * bw + i * gap + 0.02
    xb = xa + gap - 0.04
    ya = y0 + bh / 2 + 0.12
    top = ya + s / 2
    xm = xa + 0.5 * (gap - 0.04) + 0.03

    # carried-forward arrow (top-aligned with the source band)
    head_len, head_extra = 0.15, 0.05
    xh = xb - head_len
    ybot = top - r
    yc = (top + ybot) / 2
    ax.add_patch(Polygon([(xa, top), (xh, top), (xh, top + head_extra), (xb, yc),
                          (xh, ybot - head_extra), (xh, ybot), (xa, ybot)],
                         closed=True, fc=ACCENT, ec=ACCENT, lw=0.6, zorder=2))
    ax.text(xa + (xh - xa) / 2, yc, f"{n_out / n_in:.0%} kept", ha="center", va="center",
            fontsize=7, fontweight="bold", color="white", zorder=5)

    # operation label above the band
    ax.text((xa + xb) / 2, top + 0.07, op["label"], ha="center", va="bottom", fontsize=7,
            style="italic", color="#333333", linespacing=1.2)

    # dropped-rows arrow: hollow band peeling off the bottom, turning down
    y_end = y0 - 0.02
    ax.add_patch(PathPatch(drop_band(xa, ya - s / 2, d, xm, y_end),
                           fc=(*mpl.colors.to_rgb(ACCENT), 0.18), ec=ACCENT, lw=0.9, zorder=1))

    lines = op["drop"].split("\n")
    lost = f"−{(n_in - n_out) / 1e6:.2f} M rows ({(n_in - n_out) / n_in:.0%})"
    ax.text(xm, y_end - 0.05, lost, ha="center", va="top", fontsize=7.5,
            fontweight="bold", color=ink)
    ax.text(xm, y_end - 0.23, "\n".join(lines), ha="center", va="top", fontsize=6.9,
            color=muted, linespacing=1.2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=ROOT / "figs" / "plot_data_tiers_schematic_flow" / _run_timestamp())
    return p.parse_args()


args = _parse_args()
args.out.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out / "data_tiers_schematic.svg", bbox_inches="tight", pad_inches=0.03)
fig.savefig(args.out / "data_tiers_schematic.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print(f"  Saved {args.out / 'data_tiers_schematic.png'}")
update_latest(args.out.parent, args.out)
