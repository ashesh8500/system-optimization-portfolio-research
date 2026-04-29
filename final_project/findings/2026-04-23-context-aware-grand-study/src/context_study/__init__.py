"""Context-aware grand study package scaffold."""

from .analysis import build_descriptor_winner_table, compute_period_winners, summarize_features_by_winner
from .candidates import (
    CANDIDATE_DATA_DIR,
    CANDIDATE_SET_FILES,
    NAMED_CANDIDATE_SETS,
    available_candidate_sets,
    get_candidate_symbols,
    load_candidate_symbols,
    load_candidate_table,
    write_candidate_symbols,
)
from .data_loader import PriceDataCache, prepare_candidate_panel
from .descriptors import DESCRIPTOR_COLUMNS, compute_universe_descriptors
from .experiment_artifacts import (
    ExperimentArtifactWriter,
    MODEL_SELECTION_LEDGER_COLUMNS,
    TRIAL_COLUMNS,
    UNIVERSE_PROVENANCE_COLUMNS,
    build_trial_record,
    build_universe_provenance_record,
)
from .hierarchical_rl_router import (
    HierarchicalRoutingEnv,
    RouterRewardPanel,
    RouterTrainingResult,
    build_router_reward_panel,
    evaluate_router_policy,
    evaluate_routing_baselines,
    run_router_repeated_study,
    train_pufferlib_router_policy,
)
from .meta_controller import MetaController, SKLEARN_AVAILABLE, compare_meta_controller_utility
from .protocol import WalkForwardSplit, WalkForwardWindow, generate_walk_forward_splits
from .screens import (
    SCREEN_OUTPUT_COLUMNS,
    apply_screen,
    cluster_capped_momentum_screen,
    liquidity_adjusted_momentum_screen,
    low_volatility_screen,
    momentum_screen,
    volatility_adjusted_momentum_screen,
)
from .universe import (
    MEMBERSHIP_COLUMNS,
    ROLLING_UNIVERSE_TARGETS,
    STATIC_UNIVERSES,
    build_universe_membership,
)

__all__ = [
    "build_descriptor_winner_table",
    "compute_period_winners",
    "summarize_features_by_winner",
    "CANDIDATE_DATA_DIR",
    "CANDIDATE_SET_FILES",
    "NAMED_CANDIDATE_SETS",
    "available_candidate_sets",
    "get_candidate_symbols",
    "load_candidate_symbols",
    "load_candidate_table",
    "write_candidate_symbols",
    "PriceDataCache",
    "prepare_candidate_panel",
    "DESCRIPTOR_COLUMNS",
    "compute_universe_descriptors",
    "ExperimentArtifactWriter",
    "MODEL_SELECTION_LEDGER_COLUMNS",
    "TRIAL_COLUMNS",
    "UNIVERSE_PROVENANCE_COLUMNS",
    "build_trial_record",
    "build_universe_provenance_record",
    "HierarchicalRoutingEnv",
    "RouterRewardPanel",
    "RouterTrainingResult",
    "build_router_reward_panel",
    "evaluate_router_policy",
    "evaluate_routing_baselines",
    "run_router_repeated_study",
    "train_pufferlib_router_policy",
    "MetaController",
    "SKLEARN_AVAILABLE",
    "compare_meta_controller_utility",
    "WalkForwardSplit",
    "WalkForwardWindow",
    "generate_walk_forward_splits",
    "SCREEN_OUTPUT_COLUMNS",
    "apply_screen",
    "cluster_capped_momentum_screen",
    "liquidity_adjusted_momentum_screen",
    "low_volatility_screen",
    "momentum_screen",
    "volatility_adjusted_momentum_screen",
    "MEMBERSHIP_COLUMNS",
    "ROLLING_UNIVERSE_TARGETS",
    "STATIC_UNIVERSES",
    "build_universe_membership",
]
