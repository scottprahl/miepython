#!/usr/bin/env python3
"""Plot 2D near-field magnitudes for an absorbing sphere.

The plot shows both electric and magnetic near-field magnitudes on an x-z slice
through the sphere center, including points inside and outside the sphere.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from miepython.field import eh_near_cartesian


lambda0 = 1.0
d_sphere = 2.0
radius = d_sphere / 2.0
m_sphere = 1.5 - 0.1j
n_env = 1.0

extent = 2.0
npts = 121
z = np.linspace(-extent, extent, npts)
x = np.linspace(-extent, extent, npts)
Z, X = np.meshgrid(z, x, indexing="xy")
Y = np.zeros_like(X)

E_xyz, H_xyz = eh_near_cartesian(
    lambda0=lambda0,
    d_sphere=d_sphere,
    m_sphere=m_sphere,
    n_env=n_env,
    x=X,
    y=Y,
    z=Z,
    include_incident=True,
)

E_abs = np.sqrt(np.sum(np.abs(E_xyz) ** 2, axis=0))
H_abs = np.sqrt(np.sum(np.abs(H_xyz) ** 2, axis=0))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

for ax, arr, title, cmap in (
    (axes[0], E_abs, r"$|E|$", "turbo"),
    (axes[1], H_abs, r"$|H|$ (normalized)", "magma"),
):
    im = ax.imshow(
        arr,
        origin="lower",
        extent=(z.min(), z.max(), x.min(), x.max()),
        cmap=cmap,
        aspect="equal",
    )
    ax.add_patch(Circle((0.0, 0.0), radius, fill=False, color="white", lw=1.2))
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

fig.suptitle(r"Near fields for $d=2,\ m=1.5-0.1j,\ \lambda_0=1,\ n_{env}=1$")
# plt.savefig("05.svg", format="svg", bbox_inches="tight")
# plt.show()

