"""
Class to record videos from webcams using ffmpeg-python
Author: Mike Danielczuk
"""
import ffmpeg


class VideoRecorder:
    """Encapsulates video recording processes.

    Attributes
    ----------
    device_id : int
        USB index of device
    res : 2-tuple
        resolution of recording and saving. defaults to (640, 480)
    video_format : :obj:`str`
        format used for video. default to v4l2.
    fps : int
        frames per second of video captures. defaults to 30
    """

    def __init__(
        self, device_id=0, res=(640, 480), video_format="v4l2", fps=30
    ):
        self._device = device_id
        self._res = res
        self._format = video_format
        self._fps = fps

        self._recording = False

    @property
    def is_recording(self):
        return self._recording

    @property
    def is_started(self):
        return True

    def start(self):
        pass

    def start_recording(self, output_file, filters={}, overwrite=True):
        """Starts recording to a given output video file.

        Parameters
        ----------
        output_file : :obj:`str`
            filename to write video to
        filters : dict of dicts
            filters to apply to the video
        overwrite : bool
            whether to overwrite the output file if it exists
        """
        if self._recording:
            raise Exception(
                "Cannot record a video while one is already recording!"
            )
        self._recording = True
        stream = ffmpeg.input(
            "/dev/video{}".format(self._device),
            f=self._format,
            s="{}x{}".format(*self._res),
            framerate=self._fps,
        )
        for filter_name, filter_kwargs in filters.items():
            stream = ffmpeg.filter(stream, filter_name, **filter_kwargs)
        stream = ffmpeg.output(stream, output_file)
        self._video = ffmpeg.run_async(
            stream, quiet=True, overwrite_output=overwrite
        )

    def stop_recording(self):
        """Stops writing video to file."""
        if not self._recording:
            raise Exception(
                "Cannot stop a video recording when it's not recording!"
            )
        self._video.terminate()
        self._recording = False

    def stop(self):
        pass
