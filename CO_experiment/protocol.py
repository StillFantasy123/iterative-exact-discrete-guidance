"""Fixed choices used by the released Max-Cut evaluator."""

PROBLEM = "maxcut"
NUM_STATES = 2
SCHEDULE_KIND = "cosine"
SCHEDULE_EPS = 1.0e-4
GUIDANCE_STRENGTH = 1.0

__all__ = [
    "GUIDANCE_STRENGTH",
    "NUM_STATES",
    "PROBLEM",
    "SCHEDULE_KIND",
    "SCHEDULE_EPS",
]
