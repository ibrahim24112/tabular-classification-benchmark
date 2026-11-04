from __future__ import annotations

from pathlib import Path


def ensure_figure_directory(output_dir: Path | str) -> Path:
    """Create and return the directory used for selected analytical figures."""

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir
