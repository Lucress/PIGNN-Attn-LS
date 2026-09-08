"""EdgeSelfAttn model package.

Public entrypoints:
    from models.edge_selfattn import GNSMsg_EdgeSelfAttn
    from models.edge_selfattn import GNSMsg_EdgeSelfAttn_OPF

Importing this package also imports `builder`, `opf_builder`, and
`opf_supervised_builder`, which self-register all variants into
`models.registry.MODEL_REGISTRY`.
"""

from .model import GNSMsg_EdgeSelfAttn
from .opf_model import GNSMsg_EdgeSelfAttn_OPF
from . import builder               # noqa: F401  -- registers PF models
from . import opf_builder           # noqa: F401  -- registers Model 2 (physics)
from . import opf_supervised_builder  # noqa: F401  -- registers Model 1 (supervised)

__all__ = ["GNSMsg_EdgeSelfAttn", "GNSMsg_EdgeSelfAttn_OPF"]
