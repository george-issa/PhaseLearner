"""Model sketch: optical SSH model on the honeycomb lattice (paper Fig. 1).

Schematic (no data). Shows:
  * honeycomb fragment (7 complete hexagons) with A/B sublattice sites
  * the three nearest-neighbor bond vectors r_nu (nu = 1, 2, 3)
  * electron spin sigma = up/down on a site
  * the in-plane ionic displacement R_{i,gamma} = (X_{i,gamma}, Y_{i,gamma})
    of an ion in its local 2D harmonic well (frequency Omega, mass M)
  * one unit cell (dashed parallelogram, lattice vectors a1/a2) with the
    gamma = A, B basis sites

Geometry matches the DQMC source (sims/ssh/oSSH_honeycomb_mpi_execute_edited2.jl):
  a1 = (3/2,  sqrt3/2), a2 = (3/2, -sqrt3/2); A at (0,0), B at (1,0);
  r1 = (1,0), r2 = (-1/2, -sqrt3/2), r3 = (-1/2, +sqrt3/2).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

import paperstyle as ps

ps.use()

SQ3 = np.sqrt(3.0)
A1 = np.array([1.5, SQ3 / 2])
A2 = np.array([1.5, -SQ3 / 2])
R_NU = [
    np.array([1.0, 0.0]),        # r1
    np.array([-0.5, -SQ3 / 2]),  # r2
    np.array([-0.5, SQ3 / 2]),   # r3
]

BLUE, VERMILION, GREEN, ORANGE, PURPLE = ps.OKABE_ITO[:5]
R_SITE = 0.13          # site circle radius (data units; bond length = 1)
BOND_C = "0.45"
BOND_LW = 0.8


def key(p):
    return (round(p[0], 3), round(p[1], 3))


def build_flower(center_cell=(1, 1)):
    """Sites and bonds of the central hexagon + its 6 neighbors."""
    c0 = center_cell[0] * A1 + center_cell[1] * A2 + np.array([0.5, SQ3 / 2])
    ring = [np.array([SQ3 * np.cos(a), SQ3 * np.sin(a)])
            for a in np.radians([30, 90, 150, 210, 270, 330])]
    centers = [c0] + [c0 + r for r in ring]

    sites, bonds, seen_b = {}, [], set()
    for c in centers:
        verts = [c + np.array([np.cos(a), np.sin(a)])
                 for a in np.radians([0, 60, 120, 180, 240, 300])]
        for v in verts:
            sites[key(v)] = v
        for u, v in zip(verts, verts[1:] + verts[:1]):
            bk = tuple(sorted((key(u), key(v))))
            if bk not in seen_b:
                seen_b.add(bk)
                bonds.append((u, v))
    return sites, bonds


def is_A(p):
    return np.isclose(np.mod(p[0] + 0.75, 1.5), 0.75, atol=1e-3)


def draw_site(ax, p, sub, zorder=4):
    if sub == "A":
        c = Circle(p, R_SITE, facecolor=BLUE, edgecolor="black",
                   linewidth=0.35, zorder=zorder)
    else:
        c = Circle(p, R_SITE, facecolor="white", edgecolor=VERMILION,
                   linewidth=0.7, zorder=zorder)
    ax.add_patch(c)


def main():
    fig, ax = plt.subplots(figsize=(ps.ONE_COLUMN, 3.42))
    ax.set_aspect("equal")
    ax.set_axis_off()

    sites, bonds = build_flower()

    # optical-SSH bond-phonon demo: two neighbouring ions on one bond are
    # displaced in-plane from equilibrium; that bond is the modulated hopping.
    # (The B ion's other two neighbours are unit-cell vertices, so its up-right
    # A neighbour is the only partner that keeps the cell drawing clean.)
    demo_eq = np.array([4.0, 0.0])          # B ion
    demo_eq2 = np.array([4.5, SQ3 / 2])     # up-right A neighbour
    # displace each ion by > one site diameter so the solid clearly separates
    # from its ghost, in directions away from each ion's *other* bonds (B up,
    # A left) so the highlighted hopping bond stays legible.
    demo_d = np.array([0.13, 0.30])         # B in-plane displacement
    demo_d2 = np.array([-0.32, -0.12])      # A in-plane displacement
    demo_skip = (demo_eq, demo_eq2)

    for p, q in bonds:
        # the demo bond is redrawn bold between the displaced ions below
        if {key(p), key(q)} == {key(demo_eq), key(demo_eq2)}:
            continue
        ax.plot([p[0], q[0]], [p[1], q[1]], color=BOND_C, lw=BOND_LW,
                solid_capstyle="round", zorder=1)

    for p in sites.values():
        if any(np.allclose(p, s) for s in demo_skip):
            continue
        draw_site(ax, p, "A" if is_A(p) else "B")

    # ------------------------------------------------------------ unit cell
    cell0 = np.array([1.5, -SQ3 / 2])      # A corner of the shaded cell
    corners = [cell0, cell0 + A1, cell0 + A1 + A2, cell0 + A2]
    ax.add_patch(Polygon(corners, closed=True, facecolor=ORANGE,
                         alpha=0.12, edgecolor="none", zorder=0))
    # lattice vectors a1, a2 along the two cell edges
    for vec, lab, off in ((A1, r"$\mathbf{a}_1$", np.array([-0.22, 0.14])),
                          (A2, r"$\mathbf{a}_2$", np.array([-0.22, -0.14]))):
        arr = FancyArrowPatch(cell0, cell0 + vec, arrowstyle="-|>",
                              mutation_scale=6, color="black", lw=0.8,
                              shrinkA=4.5, shrinkB=4.5, zorder=3)
        ax.add_patch(arr)
        ax.annotate(lab, cell0 + 0.55 * vec + off, ha="center", va="center",
                    fontsize=9)
    ax.plot(*zip(cell0 + A1, cell0 + A1 + A2, cell0 + A2), color="black",
            lw=0.5, linestyle=(0, (4, 2)), zorder=0)
    # cell index i sits at the cell-origin (A-corner) site, not floating mid-cell
    ax.annotate(r"$\mathbf{i}$", cell0 + [0.12, 0.31],
                ha="center", va="center", fontsize=9)
    # gamma = A, B labels at the two basis sites of the cell. A sits in the
    # clear lower-left of its corner; B is pushed into the open wedge to the
    # right of its site (directly below B is blocked by the a2 diagonal and the
    # down-right bond, so the right wedge is the only collision-free pocket).
    ax.annotate(r"$\gamma = \mathrm{A}$", cell0 + [-0.30, -0.26],
                ha="center", va="top", fontsize=8)
    ax.annotate(r"$\gamma = \mathrm{B}$", cell0 + [1.50, -0.104],
                ha="center", va="center", fontsize=8)

    # ------------------------------------------------------------ r_nu arrows
    origin = np.array([3.0, SQ3])          # A site, upper left of center hex
    labels = [r"$\mathbf{r}_1$", r"$\mathbf{r}_2$", r"$\mathbf{r}_3$"]
    label_off = [np.array([0.0, 0.20]), np.array([-0.24, 0.06]),
                 np.array([-0.24, -0.06])]
    for r, lab, off in zip(R_NU, labels, label_off):
        arr = FancyArrowPatch(origin, origin + r, arrowstyle="-|>",
                              mutation_scale=7, color=GREEN, lw=1.1,
                              shrinkA=4.5, shrinkB=4.5, zorder=3)
        ax.add_patch(arr)
        ax.annotate(lab, origin + 0.55 * r + off, ha="center", va="center",
                    fontsize=9, color=GREEN)

    # ------------------------------------------------------------ spins
    # electron spin sigma on a representative site: a clean up/down arrow pair
    # (full-length shafts, proportionate heads) straddling the site centre.
    spin_site = np.array([2.5, -SQ3 / 2])  # B inside the shaded cell
    for dx, y0, y1 in ((-0.07, -0.17, 0.17), (0.07, 0.17, -0.17)):
        arr = FancyArrowPatch(spin_site + [dx, y0], spin_site + [dx, y1],
                              arrowstyle="-|>", mutation_scale=7,
                              color="black", lw=0.9, shrinkA=0, shrinkB=0,
                              zorder=6)
        ax.add_patch(arr)
    ax.annotate(r"$\sigma$", spin_site + [0.22, 0.15], ha="left",
                va="center", fontsize=9)

    # ------------------------------------------------- optical-SSH bond phonon
    # Two neighbouring ions displaced in-plane from their equilibrium "ghost"
    # positions (dashed outlines); the bond between them is the hopping
    # t_{ij}, drawn bold + coloured to show the displacements modulate it -- this
    # bond dependence is what makes the model *optical SSH* and not on-site.
    disp1, disp2 = demo_eq + demo_d, demo_eq2 + demo_d2
    ax.plot([disp1[0], disp2[0]], [disp1[1], disp2[1]], color=PURPLE,
            lw=2.0, solid_capstyle="round", zorder=2)
    for eq in demo_skip:
        ax.add_patch(Circle(eq, R_SITE, facecolor="none", edgecolor="0.55",
                            linewidth=0.5, linestyle=(0, (2, 2)), zorder=3))
    draw_site(ax, disp1, "B", zorder=5)
    draw_site(ax, disp2, "A", zorder=5)
    for eq, dp in ((demo_eq, disp1), (demo_eq2, disp2)):
        arr = FancyArrowPatch(eq, dp, arrowstyle="-|>", mutation_scale=6,
                              color="black", lw=0.9, shrinkA=0, shrinkB=4.0,
                              zorder=6)
        ax.add_patch(arr)
    # modulated-hopping label on the bold bond (offset to its open left side)
    ax.annotate(r"$t_{\mathbf{ij}}$", (disp1 + disp2) / 2 + [-0.28, 0.06],
                ha="center", va="center", fontsize=8, color=PURPLE)
    # in-plane displacement R = (X, Y), labelled just right of the demo. The
    # (X,Y) components drop the redundant i,gamma subscripts so the equation
    # ends before the B(5.5,0.866)->A(6.0,0) lattice bond (a longer label is
    # sliced by that bond -- the right hexagons leave no 1.5 du clear strip).
    ax.annotate(
        r"$\hat{\mathbf{R}}_{\mathbf{i},\gamma}=(\hat{X},\hat{Y})$",
        demo_eq + [0.40, 0.45], ha="left", va="center", fontsize=8)
    ax.annotate(r"$\Omega,\,M$", demo_eq + [0.40, 0.08], ha="left",
                va="center", fontsize=8)

    ax.set_xlim(0.8, 6.3)
    ax.set_ylim(-2.05, 3.65)
    ps.save(fig, "fig_ossh_model")


if __name__ == "__main__":
    main()
