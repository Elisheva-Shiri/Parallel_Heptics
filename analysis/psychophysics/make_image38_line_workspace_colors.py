"""Regenerate image38_line.png: the four finger panels of image38_matrix.png
laid out in a single row (1x4), with the same styling, legend and colorbar.

Reuses ``make_image38_matrix_workspace_colors.make_figure(layout="line")``.
"""
from __future__ import annotations

from make_image38_matrix_workspace_colors import make_figure


if __name__ == "__main__":
    print(make_figure(layout="line"))
