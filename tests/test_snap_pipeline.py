import unittest

from backend.snap_pipeline import SnapPipeline


class SnapPipelineSelectionTests(unittest.TestCase):
    def _pipeline_with_ranked_scores(self) -> SnapPipeline:
        pipeline = SnapPipeline.__new__(SnapPipeline)
        pipeline._score_representatives = lambda reps: [(rep, float(len(reps) - i)) for i, rep in enumerate(reps)]
        return pipeline

    def test_default_best_shot_count_is_approximately_30_percent(self) -> None:
        pipeline = self._pipeline_with_ranked_scores()
        final, ng, other = pipeline._select_buckets(list(range(40)))

        self.assertEqual(len(final), 12)
        self.assertEqual(final, list(range(12)))
        self.assertEqual(ng, [38, 39])
        self.assertEqual(other, list(range(12, 38)))

    def test_configured_best_shot_count_selects_requested_amount(self) -> None:
        pipeline = self._pipeline_with_ranked_scores()

        final_10, ng_10, other_10 = pipeline._select_buckets(list(range(40)), best_shot_count=10)
        final_30, ng_30, other_30 = pipeline._select_buckets(list(range(40)), best_shot_count=30)

        self.assertEqual(len(final_10), 10)
        self.assertEqual(final_10, list(range(10)))
        self.assertEqual(len(final_30), 30)
        self.assertEqual(final_30, list(range(30)))
        self.assertEqual(ng_10, ng_30)
        self.assertEqual(ng_10, [38, 39])
        self.assertEqual(other_10, list(range(10, 38)))
        self.assertEqual(other_30, list(range(30, 38)))

    def test_configured_best_shot_count_clamps_to_non_ng_count(self) -> None:
        pipeline = self._pipeline_with_ranked_scores()

        final, ng, other = pipeline._select_buckets(list(range(40)), best_shot_count=100)

        self.assertEqual(final, list(range(38)))
        self.assertEqual(ng, [38, 39])
        self.assertEqual(other, [])

    def test_32_representatives_default_to_10_with_fixed_ng_tail(self) -> None:
        pipeline = self._pipeline_with_ranked_scores()

        final, ng, other = pipeline._select_buckets(list(range(32)))

        self.assertEqual(len(final), 10)
        self.assertEqual(ng, [30, 31])
        self.assertEqual(other, list(range(10, 30)))

    def test_invalid_best_shot_count_raises_validation_error(self) -> None:
        pipeline = self._pipeline_with_ranked_scores()

        for invalid_value in [0, -1, 1.5, "abc"]:
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    pipeline._select_buckets(list(range(40)), best_shot_count=invalid_value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
