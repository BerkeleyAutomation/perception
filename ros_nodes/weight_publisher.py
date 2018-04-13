#!/usr/bin/env python
"""Publisher node for weight sensor.
"""
import glob
import os
import rospy
import serial
import time

from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

class WeightPublisher(object):
    """Publisher ROS node for weight sensors.

    Topics
    ------
    weights : Float32MultiArray
        The weights from each of the load cell sensors in grams,
        listed in order of ID.

    Services
    --------
    tare : Empty
        Zeros the scale at the current load.
    """

    def __init__(self, rate=20.0, id_mask='F1804'):
        """Initialize the weight publisher.

        Parameters
        ----------
        id_mask : str
            A template for the first n digits of the device IDs for valid load cells.
        """
        self._rate = rospy.Rate(rate)
        self._pub = rospy.Publisher('~weights', Float32MultiArray, queue_size=10)

        rospy.loginfo('Connecting serial')

        self._serials = self._connect(id_mask)
        if len(self._serials) == 0:
            raise ValueError('Error -- No loadstar weight sensors connected to machine!')

        # Tare the sensor
        rospy.loginfo('Tareing')
        self._tare()

        # Flush the sensor's communications
        self._flush()

        # Set up Tare service
        self._tare_service = rospy.Service('~tare', Empty, self._handle_tare)

        # Main loop -- read and publish
        while not rospy.is_shutdown():
            weights = self._read_weights()
            self._pub.publish(Float32MultiArray(data=weights))
            self._rate.sleep()


    def _handle_tare(self, request):
        """Handler for tare service.
        """
        self._tare()
        return []


    def _connect(self, id_mask):
        """Connects to all of the load cells serially.
        """
        # Get all devices attached as USB serial
        all_devices = glob.glob('/dev/ttyUSB*')

        # Identify which of the devices are LoadStar Serial Sensors
        sensors = []
        for device in all_devices:
            try:
                ser = serial.Serial(port=device,
                                    timeout=0.5,
                                    exclusive=True)
                ser.write('ID\r')
                ser.flush()
                time.sleep(0.05)
                resp = ser.read(13)
                ser.close()

                if len(resp) >= 10 and resp[:len(id_mask)] == id_mask:
                    sensors.append((device, resp.rstrip('\r\n')))
            except:
                continue
        sensors = sorted(sensors, key=lambda x : x[1])

        # Connect to each of the serial devices
        serials = []
        for device, key in sensors:
            ser = serial.Serial(port=device, timeout=0.5)
            serials.append(ser)
            rospy.loginfo('Connected to load cell {} at {}'.format(key, device))
        return serials


    def _flush(self):
        """Flushes all of the serial ports.
        """
        for ser in self._serials:
            ser.flush()
            ser.flushInput()
            ser.flushOutput()
        time.sleep(0.02)


    def _tare(self):
        """Zeros out (tare) all of the load cells.
        """
        for ser in self._serials:
            ser.write('TARE\r')
            ser.flush()
            ser.flushInput()
            ser.flushOutput()
        time.sleep(0.02)


    def _read_weights(self):
        """Reads weights from each of the load cells.
        """
        weights = []

        grams_per_pound = 453.592

        # Read from each of the sensors
        for ser in self._serials:
            ser.write('W\r')
            ser.flush()
        time.sleep(0.02)
        for ser in self._serials:
            try:
                output_str = ser.readline()
                weight = float(output_str) * grams_per_pound
                weights.append(weight)
            except:
                weights.append(0.0)

        # Log the output
        log_output = ''
        for w in weights:
            log_output +='{:.2f} '.format(w)
        rospy.loginfo(log_output)

        return weights

if __name__ == '__main__':
    try:
        rospy.init_node('weight_sensor')
        id_mask = rospy.get_param('~id_mask', 'F1804')
        rate = rospy.get_param('~rate', 20.0)
        rospy.loginfo('Starting')
        WeightPublisher(rate, id_mask)
    except rospy.ROSInterruptException:
        pass
