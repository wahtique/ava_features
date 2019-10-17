#!/usr/bin/env python
"""
Keep only what we want in the videos from the AVA dataset.

The full movies from the AVA dataset weight tens of Go. Those functions aims
to reduce the amount of data stored.

Author : William Veal Phan
Creation date : 2019-10-16T11:22
"""
from typing import Union, List

import pandas as pd
from moviepy.editor import *
from tqdm import tqdm

from AVA_features.dataset_building.misc_utils import path_to_filename_no_ext, \
    filename_to_filename_no_ext, \
    format_path


# ================== UTILS =================================


def read_labels(
        path,
        col_names: Union[List[str], None]
) -> pd.DataFrame:
    """
    Read an AVA dataset from a csv file.
    Args:
        path: str path to csv
        col_names: optional, name of columns for ease of reading

    Returns:
        read data frame
    """
    if col_names is not None:
        df = pd.read_csv(path, names=col_names)
    else:
        df = pd.read_csv(path)
    return df


def extract_video_label(labels: pd.DataFrame, video_id: str) -> pd.DataFrame:
    """
    Get the labels pertaining to a video from tje data frame containing the
    labels for all the videos.

    Args:
        labels: data frame containing the data for all videos in the data set
        video_id: id (filename without extension) of the video concerned

    Returns:
        a data frame containing the labels of the video with the given ID
    """
    return labels[labels.video_id == video_id]


def write_video_subclip(
        in_file: str,
        out_file: str,
        start: int,
        end: int,
        audio: bool = False,
        fps: int = None,
        logger: Union[str, None] = 'bar'
) -> str:
    """
    Write a subclip of a given video to a file.

    Args:
        in_file: str path to input video file
        out_file: str path to output video file
        start: int
            start time of the subclip in seconds from the beginning of the
            video
        end: int
            end time of the subclip in seconds
        audio: True if the output should have audio. Default to False.
        fps: int or None
            Target subclip framerate. Specify None to keep the input
            framerate . Default is 25.
        logger: progress bar or None

    Returns:
        str path of teh written file
    """
    with VideoFileClip(in_file).subclip(start, end) as v:
        v.write_videofile(out_file, audio=audio, fps=fps, logger=logger)
    return out_file


def process_video(
        in_path: str,
        out_path: str,
        labels: pd.DataFrame,
        audio: bool = False,
        fps: int = 25,
        logger: Union[str, None] = 'bar'
) -> Union[pd.DataFrame, None]:
    """
    Cut a video by keeping only the parts of interest, ie. parts with labels.

    Args:
        in_path: video input file
        out_path: new video output path
        labels: AVA dataset labels for all videos
        audio: bool
            should the audio be kept in the conversion ? Default to False.
        fps: int
            Default None, ie. we keep the default fps
        logger: 'bar' or None for no progress bar. Default is a progress bar

    Returns:
        a new data frame containing the labels for the new video with updated
        timestamp or None if the video is not labelled
    """
    video_id = path_to_filename_no_ext(in_path)
    video_labels = extract_video_label(labels, video_id)
    if video_labels.empty:
        # might happen if the video is not referenced in the given df
        # for example : AVA action has a unique file list for both train and
        # validation set but 2 different label files
        return None
    else:
        # annotations gives the timestamp of the middle frame, with actions
        # starting around 1 sc before and ending 1 sc after the middle time
        # thus we add 1 extra second before and after the extrema timestamp
        start = min(video_labels.middle_frame_timestamp) - 1
        end = max(video_labels.middle_frame_timestamp) + 1
        write_video_subclip(
                in_path, out_path, start, end, audio, fps, logger=logger
        )
        # update the labels to take into account that we use only the "useful"
        # part of the original video
        video_labels.middle_frame_timestamp -= start
        # gives back the updated labels
        return video_labels


def process_dir(
        in_video_dir: str,
        out_video_dir: str,
        in_label_file: str,
        out_label_dir: str,
        audio: bool = False,
        fps: int = None,
        col_names: Union[List[str], None] = None
) -> None:
    # make sure there won't b any problem with fil paths
    in_video_dir = format_path(in_video_dir)
    out_video_dir = format_path(out_video_dir)
    out_label_dir = format_path(out_label_dir)
    # get the list of video files in the given directory
    video_list = os.listdir(in_video_dir)
    # and corresponding labels
    labels = read_labels(in_label_file, col_names)

    # ffmpeg has some problem with some video filenames namely videos
    # starting with "-". It thinks they are command line args...
    # Workaround : temp file for the conversion
    tmp_out_file = f'{out_video_dir}temp.mp4'
    # if a previous session was interrupted the temp file might still be left
    if os.path.exists(tmp_out_file):
        os.remove(tmp_out_file)
    for n in tqdm(video_list):
        # Videos are referenced by ID (filename w/o extension) in the
        # annotation files
        vid_id = filename_to_filename_no_ext(n)
        # path to input video file
        in_file = f'{in_video_dir}{n}'
        # path to output video file
        out_file = f'{out_video_dir}{vid_id}.mp4'
        # since the conversion can take very long you might need to interrupt
        # it. Better make sure we don't do twice the work
        if not os.path.exists(out_file):
            df = process_video(
                    in_file, tmp_out_file, labels, audio=audio, fps=fps,
                    logger=None
            )
            # if the video has been processed
            if df is not None:
                # we write the updated label to disk
                os.rename(tmp_out_file, out_file)
                df.to_csv(f'{out_label_dir}{vid_id}.csv')
            else:
                continue
        else:
            continue


# ================= SCRIPT ========================

# From google AVA dataset home :
#
# Each row contains an annotation for one person performing an action in an
# interval, where that annotation is associated with the middle frame.
# Different persons and multiple action labels are described in separate
# rows.
#
# The format of a row is the following: video_id, middle_frame_timestamp,
# person_box, action_id, person_id
#  - video_id: YouTube identifier
#  - middle_frame_timestamp: in seconds from the start of the YouTube.
#  - person_box: top-left (x1, y1) and bottom-right (x2,y2) normalized with
# respect to frame size, where (0.0, 0.0) corresponds to the top left,
# and (1.0, 1.0) corresponds to bottom right.
#  - action_id: identifier of an action class,
#  see ava_action_list_v2.2.pbtxt
#  - person_id: a unique integer allowing this box to be linked to other
#  boxes
# depicting the same person in adjacent frames of this video.
action_labels_cols = [
        'video_id',
        'middle_frame_timestamp',
        'x1', 'y1',
        'x2', 'y2',
        'action_id',
        'person_id'
]

video_dir = 'data/videos/AVA_action/train_val'
output = 'data/videos/AVA_action/val/videos'
label_file = 'data/ava_v2.2/ava_val_v2.2.csv'
label_dir = 'data/videos/AVA_action/val/labels'

# process_dir(
#         video_dir,
#         output,
#         label_file,
#         label_dir,
#         audio=True,
#         fps=25,
#         col_names=action_labels_cols)
