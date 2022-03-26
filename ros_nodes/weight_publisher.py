#!/usr/bin/env python
"""Publisher node for weight sensor.
"""
import rospy
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

from perception import WeightSensor


class WeightPublisher(WeightSensor):
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

    def __init__(self, rate=20.0, id_mask="F1804", ntaps=4):
        """Initialize the weight publisher.

        Parameters
        ----------
        rate : float
            Rate at which weight messages are published
        id_mask : str
            A template for the first n digits of the device IDs
            for valid load cells.
        ntaps : int
            Maximum number of samples to perform filtering over.
        """
        super().__init__(id_mask, ntaps, log=False)
        self._rate = rospy.Rate(rate)
        self._pub = rospy.Publisher(
            "~weights", Float32MultiArray, queue_size=10
        )

        rospy.loginfo("Connecting to the Weight Sensor")
        self.start()

        # Tare the sensor
        rospy.loginfo("Taring")
        self.tare()

        # Set up tare service
        self._tare_service = rospy.Service("~tare", Empty, self._handle_tare)

        # Main loop -- read and publish
        while not rospy.is_shutdown():
            self._pub.publish(Float32MultiArray(data=self.read()))
            self._rate.sleep()

    def _handle_tare(self, request):
        """Handler for tare service."""
        self.tare()
        return []


if __name__ == "__main__":
    try:
        rospy.init_node("weight_sensor")
        rate = rospy.get_param("~rate", 20.0)
        id_mask = rospy.get_param("~id_mask", "F1804")
        ntaps = rospy.get_param("~ntaps", 4)
        rospy.loginfo("Starting")
        WeightPublisher(rate, id_mask, ntaps)
    except rospy.ROSInterruptException:
        pass
