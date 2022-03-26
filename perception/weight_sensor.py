"""Wrapper class for weight sensor."""
import glob
import threading
import time

import numpy as np
import serial
from autolab_core import Logger
from scipy import signal

from .constants import LBS_TO_GRAMS


class WeightSensor(object):
    """Driver for weight sensors."""

    def __init__(self, id_mask="F1804", ntaps=4, log=True):
        """Initialize the weight sensor.

        Parameters
        ----------
        id_mask : str
            A template for the first n digits of the device IDs
            for valid load cells.
        ntaps : int
            Maximum number of samples to perform filtering over.
        log : bool
            Use a logger
        """
        self._id_mask = id_mask
        self._ntaps = ntaps
        self._filter_coeffs = signal.firwin(ntaps, 0.1)
        self._running = False
        self._cur_weights = None
        self._read_thread = None
        self._write_lock = threading.Condition()
        self.logger = Logger.get_logger("WeightSensor") if log else None

    def start(self):
        """Start the sensor
        (connect and start thread for reading weight values)
        """

        if self._running:
            return
        self._serials = self._connect(self._id_mask)
        if len(self._serials) == 0:
            raise ValueError(
                "Error -- No loadstar weight sensors connected to machine!"
            )

        # Flush the sensor's communications
        self._flush()
        self._running = True

        # Start thread for reading weight sensor
        self._read_thread = threading.Thread(
            target=self._read_weights, daemon=True
        )
        self._read_thread.start()

    def stop(self):
        """Stop the sensor."""
        if not self._running:
            return
        self._running = False
        self._read_thread.join()
        self._serials = None
        self._read_thread = None

    def reset(self):
        """Starts and stops the sensor"""
        self.stop()
        self.start()

    def _connect(self, id_mask):
        """Connects to all of the load cells serially."""
        # Get all devices attached as USB serial
        all_devices = glob.glob("/dev/ttyUSB*")

        # Identify which of the devices are LoadStar Serial Sensors
        sensors = []
        for device in all_devices:
            try:
                ser = serial.Serial(port=device, timeout=0.5, exclusive=True)
                ser.write("ID\r".encode())
                time.sleep(0.05)
                resp = ser.read(13).decode()
                ser.close()

                if len(resp) >= 10 and resp[: len(id_mask)] == id_mask:
                    sensors.append((device, resp.rstrip("\r\n")))
            except (serial.SerialException, serial.SerialTimeoutException):
                continue
        sensors = sorted(sensors, key=lambda x: x[1])

        # Connect to each of the serial devices
        serials = []
        for device, key in sensors:
            ser = serial.Serial(port=device, timeout=0.5)
            serials.append(ser)
            if self.logger is not None:
                self.logger.info(
                    "Connected to load cell {} at {}".format(key, device)
                )
        return serials

    def _flush(self):
        """Flushes all of the serial ports."""
        for ser in self._serials:
            ser.flush()
            ser.flushInput()
            ser.flushOutput()
        time.sleep(0.02)

    def tare(self):
        """Zeros out (tare) all of the load cells."""
        with self._write_lock:
            self._write_lock.wait()
            for ser in self._serials:
                ser.write("TARE\r".encode())
                ser.flush()
                ser.flushInput()
                ser.flushOutput()
            time.sleep(0.02)
        if self.logger is not None:
            self.logger.info("Tared sensor")

    def read(self):
        if not self._running:
            raise ValueError("Weight sensor is not running!")
        while self._cur_weights is None:
            pass
        return self._cur_weights

    def _read_weights(self):
        weights_buffer = []
        while self._running:
            with self._write_lock:
                if len(weights_buffer) == self._ntaps:
                    weights_buffer.pop(0)
                weights_buffer.append(self._raw_weights())
                if len(weights_buffer) < self._ntaps:
                    self._cur_weights = np.mean(weights_buffer, axis=0)
                else:
                    self._cur_weights = self._filter_coeffs.dot(weights_buffer)
                self._write_lock.notify()
            time.sleep(0.005)

    def _raw_weights(self):
        """Reads weights from each of the load cells."""
        weights = []

        # Read from each of the sensors
        for ser in self._serials:
            ser.write("W\r".encode())
            ser.flush()
        time.sleep(0.02)
        for ser in self._serials:
            try:
                output_str = ser.readline().decode()
                weight = float(output_str) * LBS_TO_GRAMS
                weights.append(weight)
            except (serial.SerialException, ValueError):
                weights.append(0.0)

        # Log the output
        if self.logger is not None:
            log_output = ""
            for w in weights:
                log_output += "{:.2f} ".format(w)
            self.logger.debug(log_output)

        return weights

    def __del__(self):
        self.stop()
