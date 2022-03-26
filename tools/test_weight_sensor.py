"""
Test the loadstar weight sensor
Author: Jeff Mahler
"""
import time

from perception import WeightSensor

if __name__ == "__main__":
    # sensor
    weight_sensor = WeightSensor()
    weight_sensor.start()
    weight_sensor.tare()
    for i in range(100):
        print("Total weight:", weight_sensor.read().sum())
        time.sleep(0.05)
    weight_sensor.stop()
