"""Select and expose the active low-level backend implementation.

This module centralizes JIT vs no-JIT backend selection so other modules can
import backend callables without importing ``miepython`` package root.
"""

import os

USE_JIT = os.environ.get("MIEPYTHON_USE_JIT", "0") == "1"

if USE_JIT:
    from .mie_jit import _D_calc_nb as D_calc
    from .mie_jit import _D_downwards
    from .mie_jit import _D_upwards
    from .mie_jit import _Lentz_Dn
    from .mie_jit import _S1_S2_nb as _S1_S2
    from .mie_jit import _an_bn_nb as an_bn
    from .mie_jit import _cn_dn_nb as cn_dn
    from .mie_jit import _pi_tau_nb as pi_tau
    from .mie_jit import _single_sphere_nb as single_sphere
    from .mie_jit import _small_conducting_sphere_nb as small_conducting_sphere
    from .mie_jit import _small_sphere_nb as small_sphere
else:
    from .mie_nojit import _D_calc_py as D_calc
    from .mie_nojit import _D_downwards
    from .mie_nojit import _D_upwards
    from .mie_nojit import _Lentz_Dn
    from .mie_nojit import _S1_S2_py as _S1_S2
    from .mie_nojit import _an_bn_py as an_bn
    from .mie_nojit import _cn_dn_py as cn_dn
    from .mie_nojit import _pi_tau_py as pi_tau
    from .mie_nojit import _single_sphere_py as single_sphere
    from .mie_nojit import _small_conducting_sphere_py as small_conducting_sphere
    from .mie_nojit import _small_sphere_py as small_sphere

__all__ = (
    "USE_JIT",
    "D_calc",
    "an_bn",
    "cn_dn",
    "pi_tau",
    "single_sphere",
    "small_sphere",
    "small_conducting_sphere",
    "_S1_S2",
    "_Lentz_Dn",
    "_D_upwards",
    "_D_downwards",
)
