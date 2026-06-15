import os
import tempfile
import unittest

import numpy as np

from puppy.fusion.text_score_head import TextScoreHead


class TextScoreHeadTest(unittest.TestCase):
    def test_predict_proba_scores_are_probabilities(self) -> None:
        model = TextScoreHead(input_dim=2, seed=1)
        scores = model.predict_proba([[0.1, 0.2], [1.0, -1.0]])
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))

    def test_fit_learns_simple_synthetic_data(self) -> None:
        x = np.array(
            [
                [1.0, 1.0],
                [1.2, 0.8],
                [0.8, 1.2],
                [-1.0, -1.0],
                [-1.2, -0.8],
                [-0.8, -1.2],
            ],
            dtype=np.float32,
        )
        y = np.array([1, 1, 1, 0, 0, 0])
        model = TextScoreHead(input_dim=2, seed=3, learning_rate=0.2, epochs=250)
        model.fit(x, y)
        accuracy = np.mean(model.predict(x) == y)
        self.assertGreaterEqual(accuracy, 0.75)

    def test_save_and_load_preserve_predictions(self) -> None:
        x = np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
        y = np.array([1, 0])
        model = TextScoreHead(input_dim=2, seed=4, learning_rate=0.2, epochs=100)
        model.fit(x, y)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "text_score_head.json")
            model.save(path)
            loaded = TextScoreHead.load(path)
            np.testing.assert_allclose(model.predict_proba(x), loaded.predict_proba(x))

    def test_dimension_mismatch_raises(self) -> None:
        model = TextScoreHead(input_dim=3)
        with self.assertRaises(ValueError):
            model.predict_proba([[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
