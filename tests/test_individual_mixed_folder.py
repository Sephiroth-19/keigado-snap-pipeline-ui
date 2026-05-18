import sys
import types
import unittest

sys.modules.setdefault(
    "mediapipe",
    types.SimpleNamespace(solutions=types.SimpleNamespace(face_mesh=types.SimpleNamespace())),
)

from backend.individual.face_grouper import ClassPhotoGroup, ScoredPhoto, StudentPhotoGroup
from backend.individual_pipeline import (
    _regroup_students_by_card_index,
    normalize_card_label_without_folder_context,
)


class IndividualMixedFolderTests(unittest.TestCase):
    def test_normalize_mixed_folder_card_labels_without_context(self) -> None:
        cases = {
            "A2": ("3A", 2, "3A_002"),
            "A11": ("3A", 11, "3A_011"),
            "D32": ("3D", 32, "3D_032"),
            "H28": ("3H", 28, "3H_028"),
            "Ｈ３２": ("3H", 32, "3H_032"),
        }

        for raw_label, expected in cases.items():
            with self.subTest(raw_label=raw_label):
                normalized = normalize_card_label_without_folder_context(raw_label)
                self.assertIsNotNone(normalized)
                self.assertEqual(normalized["class_id"], expected[0])
                self.assertEqual(normalized["student_number"], expected[1])
                self.assertEqual(normalized["normalized_label"], expected[2])

    def test_invalid_mixed_folder_card_labels_are_rejected(self) -> None:
        for raw_label in ["", "32", "3A32", "A", "先生", "A1.5"]:
            with self.subTest(raw_label=raw_label):
                self.assertIsNone(normalize_card_label_without_folder_context(raw_label))

    def test_regroup_students_by_mixed_folder_card_index(self) -> None:
        student_a = StudentPhotoGroup(portraits=[ScoredPhoto(path="/tmp/a_portrait.jpg")])
        student_unknown = StudentPhotoGroup(portraits=[ScoredPhoto(path="/tmp/unknown.jpg")])
        source = ClassPhotoGroup(class_label="unknown", students=[student_a, student_unknown])

        regrouped = _regroup_students_by_card_index(
            {"unknown": source},
            {
                "/tmp/a_portrait.jpg": {
                    "accepted": True,
                    "normalized_class_id": "3A",
                    "normalized_student_number": 2,
                }
            },
        )

        self.assertIn("3A", regrouped)
        self.assertIn("UNK", regrouped)
        self.assertEqual(regrouped["3A"].students[0].attendance_number, 2)
        self.assertIsNone(regrouped["UNK"].students[0].attendance_number)


if __name__ == "__main__":
    unittest.main()
