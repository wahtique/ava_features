#!/usr/bin/env python
"""
Load videos


Author : William Veal Phan
Creation date : 2019-10-14-14-15
"""
from typing import Iterable, Union, List, Tuple

import numpy as np

from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT, imshow, waitKey, \
    destroyAllWindows, resize, INTER_AREA


def load(path: str) -> VideoCapture:
    """
    Import a video from its path
    Args:
        path: str = location of the video file

    Returns:
        cv2.VideoCapture
    """
    cap = VideoCapture(path)
    return cap


def gen_video_slices(vc: VideoCapture, l: int = 30,
                     framerate: int = 25,
                     target_size: Union[None, Tuple] = None) -> np.ndarray:
    """
    Generator yielding L sc clips of a given video capture

    Args:
        vc: VideoCapture
        l: duration in seconds f each video slice. Default = 30
        framerate: video framerate. Default = 25
        target_size: None or target size for frame resizing.
            The tuple should be : (width, height)

    Returns:
        ndarray of frames
    """
    assert l > 0
    assert vc is not None
    assert framerate > 0
    last_frame = vc.get(CAP_PROP_FRAME_COUNT) - 1
    slice_n_frames = l * framerate
    superfluous_frames = last_frame % slice_n_frames
    last_frame -= superfluous_frames
    start_last_slice = last_frame - slice_n_frames
    for i in range(0, int(start_last_slice), int(slice_n_frames)):
        frames = []
        for j in range(0, int(slice_n_frames)):
            ret, frame = vc.read()
            if target_size is not None:
                frame = resize(frame, target_size, interpolation=INTER_AREA)
            frames.append(frame)
        frames = np.array(frames)
        yield frames


def gen_thirty(path, target_size: Union[None, Tuple] = (299, 299)
               ) -> Iterable[np.ndarray]:
    """
    Wrap the creation of a generator on a video yielding 30 sc clips.

    The video is expected to have a framerate of 25 fps.

    Args:
        path: path to video file
        target_size: resizing target. None or Tuple (width, height)

    Returns:
        generator of 30 sc slices as ndarray

    """
    v = load(path)
    return gen_video_slices(v, target_size=target_size)


def play_video(capture: VideoCapture) -> None:
    """
    Play a video. For testing purpose only.


    Args:
        capture: VideoCapture to play

    Returns:
        None

    """
    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            imshow('Frame', frame)
            # this doesnt stop anything...
            if waitKey(1) & ord('q') == 0xFF:
                break
        else:
            break
    capture.release()
    destroyAllWindows()
