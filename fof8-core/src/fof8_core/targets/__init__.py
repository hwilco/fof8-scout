"""Public target package exports and registry access points."""

from fof8_core.targets.approximate_value import get_career_value_metrics
from fof8_core.targets.awards import get_awards
from fof8_core.targets.career import get_career_outcomes
from fof8_core.targets.composite import get_dpo_targets
from fof8_core.targets.draft_outcomes import get_draft_outcome_targets
from fof8_core.targets.economic import get_economic_targets
from fof8_core.targets.financial import get_annual_financials, get_merit_cap_share
from fof8_core.targets.registry import TARGET_REGISTRY, get_target, register_target
from fof8_core.targets.talent import get_peak_overall

__all__ = [
    "TARGET_REGISTRY",
    "get_target",
    "register_target",
    "get_career_outcomes",
    "get_economic_targets",
    "get_dpo_targets",
    "get_draft_outcome_targets",
    "get_annual_financials",
    "get_peak_overall",
    "get_merit_cap_share",
    "get_awards",
    "get_career_value_metrics",
]
