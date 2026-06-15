import datetime as dt
import unittest

from puppy.common.schemas import QuantFeatureTable, TextFeature
from puppy.fusion.dataset import build_text_score_dataset


def _feature(symbol: str, date: dt.date, values: list[float]) -> TextFeature:
    return TextFeature(
        agent_id=f"agent_{symbol.lower()}",
        symbol=symbol,
        date=date,
        h_text=values,
        raw_context="ctx",
        embedding_model="fake",
        dim=len(values),
    )


class TextScoreDatasetTest(unittest.TestCase):
    def test_build_dataset_joins_by_date_and_symbol(self) -> None:
        date = dt.date(2026, 1, 2)
        text_features = [
            _feature("TSLA", date, [1.0, 0.0]),
            _feature("NVDA", date, [-1.0, 0.0]),
            _feature("AAPL", date, [0.0, 1.0]),
            _feature("MSFT", date, [0.5, 0.5]),
        ]
        quant_tables = [
            QuantFeatureTable(
                date=date,
                symbols=["TSLA", "NVDA", "AAPL"],
                action_signals={"TSLA": 1, "NVDA": -1, "AAPL": 0},
                returns={"TSLA": 0.01, "NVDA": -0.01, "AAPL": 0.0},
                labels={"TSLA": 1, "NVDA": -1, "AAPL": 0},
            )
        ]
        x, y, metadata = build_text_score_dataset(text_features, quant_tables)

        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(y.tolist(), [1, 0, 0])
        self.assertEqual([item["symbol"] for item in metadata], ["TSLA", "NVDA", "AAPL"])

    def test_missing_or_none_label_is_skipped(self) -> None:
        date = dt.date(2026, 1, 2)
        text_features = [_feature("TSLA", date, [1.0]), _feature("NVDA", date, [-1.0])]
        quant_tables = [
            QuantFeatureTable(
                date=date,
                symbols=["TSLA", "NVDA"],
                action_signals={"TSLA": 1, "NVDA": -1},
                returns={"TSLA": 0.01, "NVDA": None},
                labels={"TSLA": 1, "NVDA": None},
            )
        ]
        x, y, metadata = build_text_score_dataset(text_features, quant_tables)
        self.assertEqual(x.shape, (1, 1))
        self.assertEqual(y.tolist(), [1])
        self.assertEqual(metadata[0]["symbol"], "TSLA")

    def test_dimension_mismatch_raises(self) -> None:
        date = dt.date(2026, 1, 2)
        text_features = [
            _feature("TSLA", date, [1.0, 0.0]),
            _feature("NVDA", date, [-1.0, 0.0, 0.5]),
        ]
        quant_tables = [
            QuantFeatureTable(
                date=date,
                symbols=["TSLA", "NVDA"],
                action_signals={"TSLA": 1, "NVDA": -1},
                returns={"TSLA": 0.01, "NVDA": -0.01},
                labels={"TSLA": 1, "NVDA": -1},
            )
        ]
        with self.assertRaises(ValueError):
            build_text_score_dataset(text_features, quant_tables)


if __name__ == "__main__":
    unittest.main()
