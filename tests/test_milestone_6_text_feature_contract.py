import datetime as dt
import unittest

from puppy.common.schemas import TextFeature
from puppy.fusion.text_feature_contract import (
    TextFeatureContractError,
    build_text_feature_matrix,
    validate_text_feature_batch,
)


class Milestone6TextFeatureContractTest(unittest.TestCase):
    def test_text_feature_batch_contract_and_matrix_are_stable(self) -> None:
        features = [
            _feature("TSLA", dt.date(2026, 1, 3), [0.1, 0.2, 0.3]),
            _feature("NVDA", dt.date(2026, 1, 2), [0.4, 0.5, 0.6]),
        ]

        summary = validate_text_feature_batch(features)
        matrix, metadata = build_text_feature_matrix(features)

        self.assertEqual(summary["num_features"], 2)
        self.assertEqual(summary["dim"], 3)
        self.assertEqual(summary["dtype"], "float32")
        self.assertEqual(matrix.dtype.name, "float32")
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(
            [(item["date"], item["symbol"]) for item in metadata],
            [("2026-01-02", "NVDA"), ("2026-01-03", "TSLA")],
        )

    def test_text_feature_contract_rejects_unstable_artifacts(self) -> None:
        with self.assertRaises(TextFeatureContractError):
            validate_text_feature_batch(
                [
                    _feature("TSLA", dt.date(2026, 1, 2), [0.1, 0.2]),
                    _feature("NVDA", dt.date(2026, 1, 2), [0.1, 0.2, 0.3]),
                ]
            )

        with self.assertRaises(TextFeatureContractError):
            validate_text_feature_batch(
                [
                    _feature("TSLA", dt.date(2026, 1, 2), [0.1, 0.2], model="a"),
                    _feature("NVDA", dt.date(2026, 1, 2), [0.3, 0.4], model="b"),
                ]
            )

        with self.assertRaises(TextFeatureContractError):
            validate_text_feature_batch(
                [_feature("TSLA", dt.date(2026, 1, 2), [float("nan"), 0.2])]
            )


def _feature(
    symbol: str,
    date: dt.date,
    h_text: list[float],
    model: str = "stable-encoder",
) -> TextFeature:
    return TextFeature(
        agent_id=f"sector_{symbol.lower()}",
        symbol=symbol,
        date=date,
        h_text=h_text,
        raw_context=f"Context for {symbol}",
        embedding_model=model,
        dim=len(h_text),
        dtype="float32",
    )


if __name__ == "__main__":
    unittest.main()
