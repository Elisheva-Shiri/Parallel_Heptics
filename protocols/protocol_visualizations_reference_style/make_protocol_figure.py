"""Compact protocol figure for the ICRA paper (single-column, 3.5 in wide).

Rebuilds the protocol timeline (familiarization -> four test blocks) at the
final printed size so that all text is set in real points and stays readable
at column width.  The per-participant finger-order strip was moved to the
paper text, as requested by the reviewer.

Run:  python make_protocol_figure.py
Writes image25.png (and .pdf) into the paper media folder and this folder.
"""
import random
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ----------------------------------------------------------------------------
# Geometry (inches).  Keep W/H == 1605/566 so the paper layout does not move.
W = 3.5
H = W * 566 / 1605

DPI = 460
FONT = "DejaVu Sans"  # matches the other matplotlib figures in the paper
FS_TITLE, FS_LABEL, FS_SMALL = 7.0, 6.5, 6.0

# Feedback values (x10 of the display gain in mm/m) and viridis colours
VALUES = [25, 40, 55, 70, 85, 100, 115, 130, 145]
CMAP = plt.get_cmap("viridis", len(VALUES))
COLOR = {v: CMAP(i) for i, v in enumerate(VALUES)}
STANDARD = 85

# Comparison values shown on the cards (front card first)
FAM_STACKS = [[70, 40, 130], [40, 130, 70], [130, 70, 40], [70, 130, 40]]
BLOCK_STACKS = [
    [25, 145, 55, 115, 70, 100, 40, 130],
    [25, 130, 40, 100, 70, 115, 55, 145],
    [25, 115, 70, 145, 40, 130, 55, 100],
    [25, 100, 55, 130, 70, 145, 40, 115],
]
BLOCK_NAMES = ["Index", "Middle", "Ring", "Pinky"]

# Standard/comparison presentation order is balanced within a block: on half of
# the cards the standard is shown first (drawn on top), on the other half the
# comparison is.  Fixed seed keeps the figure reproducible.
_RNG = random.Random(14)


def std_on_top(n):
    """Balanced True/False flags for ``n`` cards, in shuffled order."""
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    _RNG.shuffle(flags)
    return flags
# Finger colours as in the previous version of the figure (matplotlib tab10)
FINGER_COLOR = {"Index": "#1f77b4", "Middle": "#ff7f0e",
                "Ring": "#2ca02c", "Pinky": "#d62728"}


def card_stack(ax, x, y, values, cw, ch, dx, dy):
    """Draw a stack of two-tone cards; ``values[0]`` is the front card."""
    n = len(values)
    tops = std_on_top(n)
    for k, v in reversed(list(enumerate(values))):  # back cards first
        cx, cy = x - k * dx, y + k * dy
        lower, upper = (v, STANDARD) if tops[k] else (STANDARD, v)
        ax.add_patch(Rectangle((cx, cy), cw, ch / 2, fc=COLOR[lower],
                               ec="black", lw=0.35, zorder=n - k))
        ax.add_patch(Rectangle((cx, cy + ch / 2), cw, ch / 2, fc=COLOR[upper],
                               ec="black", lw=0.35, zorder=n - k))


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- trial axis geometry ------------------------------------------------
    y0 = 0.40
    x_start, x_fam_end, x_test_start, x_end = 0.10, 0.60, 0.72, 2.44
    x_fam_end += 50 / DPI  # trial-12 tick and label sit 50 px right of the stack edge
    n_blocks = 4
    blk_w = (x_end - x_test_start) / n_blocks
    blk_edges = [x_test_start + i * blk_w for i in range(n_blocks + 1)]
    blk_centers = [e + blk_w / 2 for e in blk_edges[:-1]]

    # ---- familiarization stacks -------------------------------------------
    fam_x = [0.12, 0.28, 0.44, 0.60]
    for x, vals in zip(fam_x, FAM_STACKS):
        card_stack(ax, x, 0.55, vals, cw=0.095, ch=0.36, dx=0.014, dy=0.010)

    # ---- experimental block stacks ---------------------------------------
    for xc, vals, name in zip(blk_centers, BLOCK_STACKS, BLOCK_NAMES):
        x = xc - 0.015
        card_stack(ax, x, 0.51, vals, cw=0.15, ch=0.40, dx=0.017, dy=0.012)
        ax.text(xc, 1.06, name, ha="center", va="center",
                fontsize=FS_TITLE, family=FONT, color=FINGER_COLOR[name])

    # ---- trial axis --------------------------------------------------------
    ax.plot([x_start, x_end], [y0, y0], color="black", lw=0.9,
            solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((x_end + 0.04, y0), (2.64, y0),
                                 arrowstyle="-|>", mutation_scale=6,
                                 color="black", lw=0.9))
    ax.text(2.57, y0 + 0.09, "Trial", ha="center", va="center",
            fontsize=FS_LABEL, family=FONT)

    # boundary ticks, each labelled with its trial number.  The start of the
    # first test block (trial 13) is not marked: the gap after trial 12 shows it.
    ticks = [(x_start, "1"), (x_fam_end, "12")]
    ticks += [(e, str(12 + 64 * (i + 1))) for i, e in enumerate(blk_edges[1:])]
    for xt, lab in ticks:
        ax.plot([xt, xt], [y0 - 0.035, y0 + 0.035], color="black", lw=0.9)
        ax.text(xt, y0 - 0.10, lab, ha="center", va="center",
                fontsize=FS_LABEL, family=FONT)

    # ---- legend: one column of swatches -----------------------------------
    lx, sw, row_h, col_w = 2.76, 0.07, 0.098, 0.31
    split = 4  # first column: the four lowest values
    for i, v in enumerate(VALUES):
        col = 0 if i < split else 1
        row = i if col == 0 else i - split
        x = lx + col * col_w
        y = 1.00 - row * row_h
        ax.add_patch(Rectangle((x, y - sw / 2), sw, sw, fc=COLOR[v],
                               ec="black", lw=0.35))
        gain = f"{v / 10:.1f}"  # legend is in display-gain units (mm/m)
        label = f"{gain} s" if v == STANDARD else gain
        ax.text(x + sw + 0.03, y, label, ha="left", va="center",
                fontsize=FS_SMALL, family=FONT)

    here = Path(__file__).resolve().parent
    root = here.parents[1]
    media = root / "paper" / "icra2027_paper" / "icra2027_latex_transfer" /         "media" / "media"

    tmp = here / "_protocol_tmp.png"
    fig.savefig(tmp, dpi=DPI, facecolor="white")
    fig.savefig(here / "protocol_compact.pdf", facecolor="white")

    # trim empty bands at top and bottom only, so the horizontal scale (and
    # therefore the printed font size at column width) stays exactly as designed
    im = Image.open(tmp)
    arr = np.asarray(im.convert("L"))
    rows = np.where((arr < 250).any(axis=1))[0]
    pad = int(0.015 * DPI)
    top = max(0, rows[0] - pad)
    bottom = min(arr.shape[0], rows[-1] + pad)
    im = im.crop((0, top, arr.shape[1], bottom))
    for out in (media / "image25.png", here / "protocol_compact.png"):
        for attempt in range(4):  # Dropbox can briefly lock the target file
            try:
                im.save(out)
                break
            except OSError:
                time.sleep(3)
    tmp.unlink()
    print("wrote", media / "image25.png")


if __name__ == "__main__":
    main()
