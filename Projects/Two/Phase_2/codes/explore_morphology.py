"""Teaching examples for L5 pyramidal-cell morphology access.

Run:
    python explore_morphology.py

The script makes ``morphology_domains.png`` and prints examples showing how to:
* tell soma, basal dendrite, apical dendrite, and axon apart;
* access a domain's sections and segments;
* select proximal or distal segments by distance from soma.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cell import HPC, h


DOMAIN_COLORS = {
    "somatic": "black",
    "basal": "tab:blue",
    "apical": "tab:orange",
    "axonal": "0.55",
}


def point_at_segment(sec, x):
    """Approximate a segment's 3D coordinate from NEURON morphology points."""

    n3d = int(h.n3d(sec=sec))
    if n3d == 0:
        return np.nan, np.nan, np.nan

    xs = np.array([h.x3d(i, sec=sec) for i in range(n3d)])
    ys = np.array([h.y3d(i, sec=sec) for i in range(n3d)])
    zs = np.array([h.z3d(i, sec=sec) for i in range(n3d)])

    if n3d == 1:
        return xs[0], ys[0], zs[0]

    step = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
    cumulative = np.concatenate([[0], np.cumsum(step)])
    target = x * cumulative[-1]
    return (
        np.interp(target, cumulative, xs),
        np.interp(target, cumulative, ys),
        np.interp(target, cumulative, zs),
    )


def plot_section(ax_xy, ax_xz, sec, color, linewidth=0.8, alpha=0.9):
    """Draw one NEURON section in two projections: x-y and x-z."""

    n3d = int(h.n3d(sec=sec))
    if n3d < 2:
        return

    xs = [h.x3d(i, sec=sec) for i in range(n3d)]
    ys = [h.y3d(i, sec=sec) for i in range(n3d)]
    zs = [h.z3d(i, sec=sec) for i in range(n3d)]

    ax_xy.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
    ax_xz.plot(xs, zs, color=color, linewidth=linewidth, alpha=alpha)


def save_domain_figure(cell, highlighted_segments, output_path):
    """Save a morphology figure colored by anatomical domain."""

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    for domain, color in DOMAIN_COLORS.items():
        for sec in cell.sections(domain):
            plot_section(ax_xy, ax_xz, sec, color)

    if highlighted_segments:
        coords = np.array([point_at_segment(seg.sec, seg.x) for seg in highlighted_segments])
        ax_xy.scatter(coords[:, 0], coords[:, 1], s=18, color="crimson", label="selected segments")
        ax_xz.scatter(coords[:, 0], coords[:, 2], s=18, color="crimson", label="selected segments")

    for ax, ylabel in [(ax_xy, "y (um)"), (ax_xz, "z (um)")]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (um)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)

    ax_xy.set_title("x-y projection")
    ax_xz.set_title("x-z projection")

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=domain)
        for domain, color in DOMAIN_COLORS.items()
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="crimson", lw=0, label="selected")
    )
    ax_xy.legend(handles=legend_handles, frameon=False, loc="best")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def describe_segment(cell, seg, label):
    """Print section name, relative x, and soma path distance for one segment."""

    print(
        f"{label}: {seg.sec.name()}({seg.x:.3f}), "
        f"distance_from_soma = {cell.distance_from_soma(seg):.1f} um"
    )


def main():
    cell = HPC("./", spine_density=0.0)

    cell.summarize_domains()

    print("\nDirect domain access examples")
    print("-----------------------------")
    print("First basal section:", cell.sections("basal")[0].name())
    print("First apical section:", cell.sections("apical")[0].name())
    describe_segment(cell, cell.HCell.apic[50](0.5), "A commonly used apical segment")

    print("\nDistance-based selection examples")
    print("---------------------------------")
    proximal_basal = cell.segments_in_distance_range("basal", min_um=0, max_um=150)
    proximal_apical = cell.segments_in_distance_range("apical", min_um=0, max_um=150)
    apical_600_800 = cell.segments_in_distance_range("apical", min_um=600, max_um=800)

    print(f"Basal dendrite segments within 150 um of soma: {len(proximal_basal)}")
    print(f"Apical dendrite segments within 150 um of soma: {len(proximal_apical)}")
    print(f"Apical dendrite segments from 600-800 um: {len(apical_600_800)}")

    describe_segment(cell, cell.nearest_segment("apical", 650), "Nearest apical segment to 650 um")

    output_path = Path("morphology_domains.png")
    save_domain_figure(cell, apical_600_800, output_path)
    print(f"\nSaved morphology figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()

