from __future__ import annotations


PROTECTED_PURPOSES = {
    "fit", "early_stopping", "model_selection", "calibration", "oof",
    "threshold_selection", "scaler_fit", "class_weighting", "hyperparameter_selection",
}


class TestSplitAccessError(RuntimeError):
    pass


class SplitAccessGuard:
    def __init__(self) -> None:
        self.access_log: list[dict[str, str]] = []

    def assert_allowed(self, split: str, purpose: str) -> None:
        normalized = str(split).lower()
        purpose = str(purpose).lower()
        self.access_log.append({"split": normalized, "purpose": purpose})
        if normalized == "test" and purpose in PROTECTED_PURPOSES:
            raise TestSplitAccessError(f"Test split access forbidden for {purpose}")

    @property
    def test_feature_access_count(self) -> int:
        return sum(item["split"] == "test" for item in self.access_log)
