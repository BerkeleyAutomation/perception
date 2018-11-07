import pyrealsense2 as rs
import matplotlib.pyplot as plt

from perception import RgbdSensorFactory


def discover_cams():
    """Returns a list of the ids of all cameras connected via USB."""
    ctx = rs.context()
    ctx_devs = list(ctx.query_devices())
    ids = []
    for i in range(ctx.devices.size()):
        ids.append(ctx_devs[i].get_info(rs.camera_info.serial_number))
    return ids


def main():
    ids = discover_cams()
    assert ids, "[!] No camera detected."

    cfg = {}
    cfg['cam_id'] = ids[0]
    cfg['filter_depth'] = True
    cfg['frame'] = 'realsense_overhead'

    sensor = RgbdSensorFactory.sensor('realsense', cfg)
    sensor.start()
    camera_intr = sensor.color_intrinsics
    color_im, depth_im, _ = sensor.frames()
    sensor.stop()

    print("intrinsics matrix: {}".format(camera_intr.K))

    fig, axes = plt.subplots(1, 2)
    for ax, im in zip(axes, [color_im.data, depth_im.data]):
        ax.imshow(im)
        ax.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
