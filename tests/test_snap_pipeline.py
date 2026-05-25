import unittest

from backend.snap_pipeline import DEFAULT_BEST_SHOT_COUNT, SnapPipeline


class SnapPipelineSelectionTests(unittest.TestCase):
    def _pipeline_with_ranked_scores(self) -> SnapPipeline:
        pipeline = SnapPipeline.__new__(SnapPipeline)
        pipeline._score_representatives = lambda reps: [(rep, float(len(reps) - i)) for i, rep in enumerate(reps)]
        return pipeline

    def test_default_best_shot_count_remains_25(self) -> None:
        pipeline = self._pipeline_with_ranked_scores()
        final, ng, other = pipeline._select_buckets(list(range(40)))

        self.assertEqual(DEFAULT_BEST_SHOT_COUNT, 25)
        self.assertEqual(len(final), 25)
        self.assertEqual(final, list(range(25)))
        self.assertEqual(len(ng) + len(other), 15)

    def test_configured_best_shot_count_selects_requested_amount(self) -> None:
        pipeline = self._pipeline_with_ranked_scores()

        final_10, _, _ = pipeline._select_buckets(list(range(40)), best_shot_count=10)
        final_30, _, _ = pipeline._select_buckets(list(range(40)), best_shot_count=30)

        self.assertEqual(len(final_10), 10)
        self.assertEqual(final_10, list(range(10)))
        self.assertEqual(len(final_30), 30)
        self.assertEqual(final_30, list(range(30)))

    def test_invalid_best_shot_count_raises_validation_error(self) -> None:
        pipeline = self._pipeline_with_ranked_scores()

        for invalid_value in [0, -1, 201, 1.5, "abc"]:
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    pipeline._select_buckets(list(range(40)), best_shot_count=invalid_value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
