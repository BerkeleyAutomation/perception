'''
Script to register an object in world frame provided transform from cam to cb and from obj to cb
Author: Jacky
'''
import IPython
import os
import argparse
import logging 

from autolab_core import RigidTransform, YamlConfig
from perception import CameraChessboardRegistration, RgbdSensorFactory

VIS_SUPPORTED = True
try:
    from meshpy import ObjFile
    from visualization import Visualizer3D as vis
except:
    print 'Failed to import AUTOLAB meshpy and / or visualization package. Visualizatoin disabled'
    VIS_SUPPORTED = FALSE

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('object_name')
    args = parser.parse_args()

    config_filename = 'cfg/tools/register_object.yaml'
    config = YamlConfig(config_filename)
    
    sensor_frame = config['sensor']['frame_name']
    sensor_type = config['sensor']['type']
    sensor_config = config['sensor']

    object_path = os.path.join(config['objects_dir'], args.object_name)
    obj_cb_transform_file_path = os.path.join(object_path, 'T_cb_{0}.tf'.format(args.object_name))

    # load T_cb_obj
    T_cb_obj = RigidTransform.load(obj_cb_transform_file_path)
    
    # load T_world_cam 
    T_world_cam_path = os.path.join(config['calib_dir'], sensor_frame, '{0}_to_world.tf'.format(sensor_frame))
    T_world_cam = RigidTransform.load(T_world_cam_path)

    # open sensor
    sensor_type = sensor_config['type']
    sensor_config['frame'] = sensor_frame
    sensor = RgbdSensorFactory.sensor(sensor_type, sensor_config)
    logging.info('Starting sensor')
    sensor.start()
    ir_intrinsics = sensor.ir_intrinsics
    logging.info('Sensor initialized')

    # register
    reg_result = CameraChessboardRegistration.register(sensor, config['chessboard_registration'])
    
    T_cb_cam = reg_result.T_camera_cb
    T_world_obj = T_world_cam * T_cb_cam.inverse() * T_cb_obj

    output_path = os.path.join(config['calib_dir'], T_world_obj.from_frame)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_filename = os.path.join(output_path, '{0}_to_world.tf'.format(T_world_obj.from_frame))
    print T_world_obj
    T_world_obj.save(output_filename)

    if config['vis'] and VIS_SUPPORTED:

        _, depth_im, _ = sensor.frames()
        pc_cam = ir_intrinsics.deproject(depth_im)
        pc_world = T_world_cam * pc_cam

        mesh_file = ObjFile(os.path.join(object_path, '{0}.obj'.format(args.object_name)))
        mesh = mesh_file.read()

        vis.figure(bgcolor=(0.7,0.7,0.7))
        vis.mesh(mesh, T_world_obj.as_frames('obj', 'world'), style='surface')
        vis.pose(T_world_obj, alpha=0.04, tube_radius=0.002, center_scale=0.01)
        vis.pose(RigidTransform(from_frame='origin'), alpha=0.04, tube_radius=0.002, center_scale=0.01)
        vis.pose(T_world_cam, alpha=0.04, tube_radius=0.002, center_scale=0.01)
        vis.pose(T_world_cam  * T_cb_cam.inverse(), alpha=0.04, tube_radius=0.002, center_scale=0.01)
        vis.points(pc_world, subsample=20)
        vis.show()
    sensor.stop()
