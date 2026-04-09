from dataclasses import dataclass
from enum import IntEnum


class LinearizationBias(IntEnum):
    """Bias strategy for graph traversal based on node connectivity.

    0: random_order        - Stochastic selection
    1: max_degree_first    - Prioritize high-connectivity nodes (Hubs)
    2: min_degree_first    - Prioritize low-connectivity nodes (Leaves)
    """

    RANDOM_ORDER = 0
    MAX_DEGREE_FIRST = 1
    MIN_DEGREE_FIRST = 2


@dataclass
class LinearizationStrategy:
    name: str
    start_bias: LinearizationBias
    jump_bias: LinearizationBias
    neighbor_bias: LinearizationBias

    @property
    def kwargs(self):
        """Returns the kwargs expected by the Cython sampler."""
        return {
            "start_bias": int(self.start_bias),
            "jump_bias": int(self.jump_bias),
            "neighbor_bias": int(self.neighbor_bias),
        }


# --- THE REGISTRY ---
STRATEGIES = {
    "random_order": LinearizationStrategy(
        "random_order",
        LinearizationBias.RANDOM_ORDER,
        LinearizationBias.RANDOM_ORDER,
        LinearizationBias.RANDOM_ORDER,
    ),
    "max_degree_first": LinearizationStrategy(
        "max_degree_first",
        LinearizationBias.MAX_DEGREE_FIRST,
        LinearizationBias.MAX_DEGREE_FIRST,
        LinearizationBias.MAX_DEGREE_FIRST,
    ),
    "min_degree_first": LinearizationStrategy(
        "min_degree_first",
        LinearizationBias.MIN_DEGREE_FIRST,
        LinearizationBias.MIN_DEGREE_FIRST,
        LinearizationBias.MIN_DEGREE_FIRST,
    ),
    "anchor_expansion": LinearizationStrategy(
        "anchor_expansion",
        LinearizationBias.MAX_DEGREE_FIRST,  # Start at Hub (Anchor)
        LinearizationBias.MAX_DEGREE_FIRST,  # Jump to Hub (Re-anchor)
        LinearizationBias.MIN_DEGREE_FIRST,  # Walk to Leaves (Expand)
    ),
}
