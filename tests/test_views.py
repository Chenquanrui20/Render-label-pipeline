"""Camera message parsing: both the full schema and the centre-only variant."""
import json
import os
import tempfile
import unittest

import numpy as np

from renderlabel import views as V


def _message(with_tvec=True):
    # a plausible camera: R_world2cam = identity, centre at (1, 2, 3)
    R = np.eye(3)
    C = np.array([1.0, 2.0, 3.0])
    d = dict(image_name="f.png", render_file="f.png",
             width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
             R_world2cam=R.tolist(), R_cam2world=R.tolist(),
             camera_center_world=C.tolist())
    if with_tvec:
        d["T_world2cam"] = (-R @ C).tolist()
    return d


class MessageTest(unittest.TestCase):

    def _load(self, message):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "message"))
            with open(os.path.join(d, "message", "f.png.json"), "w") as fh:
                json.dump(message, fh)
            return V.load_views(d)

    def test_full_schema(self):
        views = self._load(_message(with_tvec=True))
        self.assertEqual(len(views), 1)
        np.testing.assert_allclose(views[0]["t"], [-1.0, -2.0, -3.0])

    def test_missing_tvec_is_derived_from_centre(self):
        views = self._load(_message(with_tvec=False))
        np.testing.assert_allclose(views[0]["t"], [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(views[0]["C"], [1.0, 2.0, 3.0])

    def test_inconsistent_tvec_rejected(self):
        m = _message(with_tvec=True)
        m["T_world2cam"] = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            self._load(m)


if __name__ == "__main__":
    unittest.main()
