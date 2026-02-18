"""Ensemble model configuration."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


def build_ensemble(random_state: int = 42) -> VotingClassifier:
    """Build a soft-voting ensemble for fraud detection."""
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    gb = GradientBoostingClassifier(random_state=random_state)

    return VotingClassifier(
        estimators=[
            ("rf", rf),
            ("lr", lr),
            ("gb", gb),
        ],
        voting="soft",
    )
