"""
Test the loadstart weight sensor
Author: Jeff Mahler
"""
import logging
import rospy
import sys
import time

from perception import WeightSensor

if __name__ == '__main__':
    # set logging
    logging.getLogger().setLevel(logging.INFO)
    rospy.init_node('weight_sensor', anonymous=True)

    # sensor
    weight_sensor = WeightSensor()
    weight_sensor.start()    
    for i in range(10000):
        print 'Total weight:', weight_sensor.total_weight()
        time.sleep(0.1)
    weight_sensor.stop()
