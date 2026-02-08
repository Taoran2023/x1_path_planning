# agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# =========================
# Model Tags (enum-like)
# =========================
class ModelTag:
    UAV = "UAV"
    UGV = "UGV"
    LEGGED = "legged"


# =========================
# Energy model (parameters only)
# =========================
@dataclass(frozen=True)
class EnergyModel:
    """
    Parameter container ONLY.

    - model: model tag (UGV/UAV/legged)
    - energy_model_tag: selects which formula to use in tree_search.py
      e.g. "linear", "poly", "exp", "log", "power", "custom"
    - energy_factors: parameters for the chosen formula (dict; flexible)
    - region_factors: optional multiplier per region tag (e.g. indoor/outdoor/unknown)
    - score_mode/score_params/edge_type_factors: optional knobs for tree_search to interpret (demo can ignore)
    """
    model: str
    energy_model_tag: str = "linear"
    energy_factors: Dict[str, Any] = field(default_factory=dict)

    region_factors: Dict[str, float] = field(default_factory=dict)

    # Optional: allow tree_search to use these if you want; otherwise ignore in demo
    score_mode: str = "ignore"                 # "ignore" | "linear" | "log" | "custom"
    score_params: Dict[str, Any] = field(default_factory=dict)

    edge_type_factors: Dict[str, float] = field(default_factory=dict)  # {"intra":1.0,"portal":1.1,"anchor":1.0}


# =========================
# Agent (static config only)
# =========================
@dataclass
class Agent:
    """
    Static agent configuration only.
    All cost computation is done in tree_search.py.

    Notes:
      - remain_energy/remain_time/current_model are NOT used in this demo;
        keep them only for logging/debug if you want.
    """
    name: str

    # supported model tags (e.g., ["UGV","legged"] for M4 ground; ["UAV"] for drone-only)
    models: List[str] = field(default_factory=list)

    # model -> EnergyModel (parameter container)
    energy_models: Dict[str, EnergyModel] = field(default_factory=dict)

    # future: transformation cost (demo: not used)
    transform_energy: float = 0.0

    # budgets (demo uses only energy_budget)
    energy_budget: float = 100.0
    time_budget: float = 0.0  # not used in demo

    # hard-block tags for this agent (tree_search should check this)
    no_traversable_tags: Set[str] = field(default_factory=set)

    # optional record fields (not used in demo search state)
    remain_energy: Optional[float] = None
    remain_time: Optional[float] = None
    current_model: Optional[str] = None

    def __post_init__(self):
        self.models = [str(m) for m in self.models]
        if self.energy_budget <= 0:
            raise ValueError("energy_budget must be > 0")

    def supports(self, model: str) -> bool:
        return model in self.models

    def is_tag_blocked(self, node_tag: str) -> bool:
        return node_tag in self.no_traversable_tags

    def get_energy_model(self, model: str) -> EnergyModel:
        em = self.energy_models.get(model, None)
        if em is None:
            raise KeyError(f"Agent '{self.name}' has no EnergyModel for model='{model}'")
        return em
