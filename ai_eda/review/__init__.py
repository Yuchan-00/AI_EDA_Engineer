"""Independent review.

Re-checks consistency *between* stages so an error in one stage cannot
propagate silently. The reviewer shares no state with the agents that
produced the design; it only reads the IR, the artifacts on disk and the
tool reports.
"""

from ai_eda.review.areas import ReviewArea
from ai_eda.review.reviewer import IndependentReviewer, ReviewReport

__all__ = ["IndependentReviewer", "ReviewArea", "ReviewReport"]
