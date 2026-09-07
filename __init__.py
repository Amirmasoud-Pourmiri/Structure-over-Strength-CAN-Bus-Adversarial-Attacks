"""IDS family registry used by the evaluation harness."""
from ids.base import IDS
from ids.rule_based import RuleBasedIDS
from ids.statistical import StatisticalIDS
from ids.lstm_ae import LSTMAutoencoderIDS
from ids.hybrid import HybridIDS

FAMILIES = {
    "rule": RuleBasedIDS,
    "statistical": StatisticalIDS,
    "lstm_ae": LSTMAutoencoderIDS,
    "hybrid": HybridIDS,
}


def build_ids(family: str, seed: int = 0, **kw) -> IDS:
    cls = FAMILIES[family]
    try:
        return cls(seed=seed, **kw)        # ML families accept a seed
    except TypeError:
        return cls(**kw)
