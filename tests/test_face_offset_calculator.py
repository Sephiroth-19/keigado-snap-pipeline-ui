import importlib
import sys
import types
import unittest


def _install_import_stubs():
    sys.modules.setdefault(
        "cv2",
        types.SimpleNamespace(
            COLOR_BGR2RGB=1,
            IMREAD_COLOR=1,
            cvtColor=lambda img, code: img,
            imdecode=lambda buf, flags: None,
        ),
    )
    sys.modules.setdefault(
        "mediapipe",
        types.SimpleNamespace(
            solutions=types.SimpleNamespace(
                face_mesh=types.SimpleNamespace(),
                selfie_segmentation=types.SimpleNamespace(),
            )
        ),
    )


_install_import_stubs()
face_offsets = importlib.import_module("backend.individual.face_offset_calculator")


class FaceOffsetCalculatorTests(unittest.TestCase):
    def test_target_eye_distance_uses_v10_formula_and_clamp(self):
        params = face_offsets.get_target_params(
            {
                "frame_w_mm": 36.0,
                "frame_h_mm": 44.0,
                "guide_ratios": {"top_ratio": 0.10, "bottom_ratio": 0.86},
                "scale_clamp": {"min": 125, "max": 145},
            }
        )

        self.assertEqual(params["target_chin_y"], 0.86)
        self.assertEqual(params["target_top_y"], 0.10)
        self.assertEqual(params["target_eye_dist"], 0.26)
        self.assertEqual(params["scale_clamp_min"], 125.0)
        self.assertEqual(params["scale_clamp_max"], 145.0)

    def test_compute_offsets_uses_subject_top_dual_anchor_and_horizontal_centering(self):
        offsets = face_offsets.compute_offsets(
            {
                "img_width": 4000,
                "img_height": 6000,
                "face_center_x": 0.60,
                "chin_y": 0.58,
                "forehead_y": 0.30,
                "subject_top_y": 0.22,
                "eye_dist": 0.08,
            },
            frame_w_mm=36.0,
            frame_h_mm=44.0,
            target_face_center_x=0.50,
            target_chin_y=0.86,
            target_top_y=0.10,
            crown_k=0.85,
            scale_clamp_min=100,
            scale_clamp_max=200,
            target_eye_dist=0.26,
            chin_k=0.0,
        )

        self.assertEqual(offsets["method"], "selfie_segmentation_v15")
        self.assertEqual(offsets["scaleFactor"], 172.0)
        self.assertEqual(offsets["offsetY"], 8.41)
        self.assertEqual(offsets["offsetX"], -6.19)
        self.assertEqual(offsets["subject_top_y"], 0.22)

    def test_compute_offsets_falls_back_to_v10_eye_scale_without_forehead(self):
        offsets = face_offsets.compute_offsets(
            {
                "img_width": 4000,
                "img_height": 6000,
                "face_center_x": 0.50,
                "chin_y": 0.60,
                "eye_dist": 0.12,
            },
            frame_w_mm=36.0,
            frame_h_mm=44.0,
            target_chin_y=0.86,
            target_top_y=0.10,
            scale_clamp_min=125,
            scale_clamp_max=145,
            target_eye_dist=0.26,
        )

        self.assertEqual(offsets["method"], "chin_anchor_eye_scale_v10_fallback")
        self.assertEqual(offsets["scaleFactor"], 145.0)
        self.assertIn("offsetX", offsets)
        self.assertIn("offsetY", offsets)


if __name__ == "__main__":
    unittest.main()
