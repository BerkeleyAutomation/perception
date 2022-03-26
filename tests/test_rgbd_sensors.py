import os
import unittest

import numpy as np
from autolab_core import CameraIntrinsics, ColorImage, DepthImage

from perception import RgbdSensorFactory

IM_FILEROOT = "tests/data"


class TestRgbdSensors(unittest.TestCase):
    def test_virtual(self, height=100, width=100):
        # Generate folder of color and depth images
        if not os.path.exists(IM_FILEROOT):
            os.makedirs(IM_FILEROOT)
        cam_intr = CameraIntrinsics(
            "a",
            fx=0.0,
            fy=0.0,
            cx=0.0,
            cy=0.0,
            skew=0.0,
            height=100,
            width=100,
        )
        cam_intr.save(os.path.join(IM_FILEROOT, "a.intr"))
        color_data = (255 * np.random.rand(10, height, width, 3)).astype(
            np.uint8
        )
        depth_data = np.random.rand(10, height, width).astype(np.float32)
        for i in range(10):
            im = ColorImage(color_data[i], frame="a")
            im.save(os.path.join(IM_FILEROOT, "color_{:d}.png".format(i)))

            im = DepthImage(depth_data[i], frame="a")
            im.save(os.path.join(IM_FILEROOT, "depth_{:d}.npy".format(i)))

        # Create virtual camera
        virtual_cam = RgbdSensorFactory.sensor(
            "virtual", cfg={"image_dir": IM_FILEROOT, "frame": "a"}
        )
        self.assertTrue(
            virtual_cam.path_to_images == IM_FILEROOT,
            msg="img path changed after init",
        )

        # Start virtual camera and read frames
        virtual_cam.start()
        self.assertTrue(
            virtual_cam.is_running, msg="camera not running after start"
        )
        for i in range(10):
            color, depth = virtual_cam.frames()
            self.assertTrue(
                np.all(color.data == color_data[i]),
                msg="color data for img {:d} changed".format(i),
            )
            self.assertTrue(
                color.frame == virtual_cam.frame,
                msg="frame mismatch between color im and camera",
            )
            self.assertTrue(
                np.all(depth.data == depth_data[i]),
                msg="depth data for img {:d} changed".format(i),
            )
            self.assertTrue(
                depth.frame == virtual_cam.frame,
                msg="frame mismatch between depth im and camera",
            )

        # Make sure camera is stopped
        virtual_cam.stop()
        self.assertFalse(
            virtual_cam.is_running, msg="camera running after stop"
        )

        # Cleanup images
        for i in range(10):
            os.remove(os.path.join(IM_FILEROOT, "color_{:d}.png".format(i)))
            os.remove(os.path.join(IM_FILEROOT, "depth_{:d}.npy".format(i)))
        os.remove(os.path.join(IM_FILEROOT, "a.intr"))
        os.rmdir(IM_FILEROOT)


if __name__ == "__main__":
    unittest.main()
