#!/usr/bin/env python
"""
Author : William Veal Phan
Creation date : 2019-10-28T23:34
"""
import os

import cv2
import numpy as np
import multiprocessing as mp
import pandas as pd
from typing import Union, Tuple
from tensorflow.python.keras.utils.data_utils import Sequence
from pims import Video

ACTION_LABELS = [79, 80, 74]


class AVAGenerator(Sequence):
    __slots__ = (
            'batch_size', 'target_img_shape', 'target_fps', 'shuffle',
            'sequence_time_span', 'videos', 'labels', 'batches', 'nbatches'
    )

    def __init__(
            self,
            dir_path: str,
            # path to directory with videos and labels subdirectories
            batch_size: int = 8,
            sequence_time_span: int = 2,
            target_img_shape: Union[Tuple[int, int], None] = None,
            target_fps: Union[int, None] = None,
            shuffle: bool = False
    ):
        self.batch_size = batch_size
        self.target_img_shape = target_img_shape
        self.target_fps = target_fps
        self.sequence_time_span = sequence_time_span
        videos_path = f'{dir_path}/videos/'
        labels_path = f'{dir_path}/labels/'
        self.videos = tuple(f'{videos_path}{f}' for f in os.listdir(
                videos_path))  # lists of str like {video_id}.{[mp4 ,csv]}
        self.labels = tuple(
                f'{labels_path}{f}' for f in os.listdir(labels_path))
        self.batches = generate_batches(
                self.videos,
                self.labels,
                self.batch_size,
                self.sequence_time_span
        )
        self.nbatches = len(self.batches)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, index):
        b: AVAGenerator.VideoBatch = self.batches[index]
        v = Video(b.video_path)
        seqs = [s for s in b.sequences]
        xy_b = tuple(self.__sequence_to_tuple__(s, v) for s in seqs)
        x_b = tuple(x for x, _ in xy_b)
        y_b = tuple(y for _, y in xy_b)
        return np.asarray(x_b), np.asarray(y_b)

    def __sequence_to_tuple__(self, seq, v: Video):
        if self.target_fps is not None:
            sample = np.random.choice(
                    range(seq.start, seq.end),
                    self.target_fps * self.sequence_time_span
            )
            x = v[sample]
        else:
            x = v[seq.start:seq.end]
        if self.target_img_shape is not None:
            x = np.asarray(
                    tuple(
                            cv2.resize(f, self.target_img_shape) for f in x
                    )
            )
        else:
            x = np.asarray(x)
        y = seq.label
        return x, y

    def __len__(self):
        return self.nbatches

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batches)


class VideoBatch:
    __slots__ = ('video_path', 'sequences')

    def __init__(self, video_path, sequences):
        self.video_path = video_path
        self.sequences = sequences


class VideoSeq:
    __slots__ = ('start', 'end', 'label')

    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label


def generate_batches(videos, labels, batch_size, sequence_time_span):
    pool = mp.Pool(mp.cpu_count())
    batches = pool.starmap(
            build_video_batch,
            [
                    (v, l, batch_size, sequence_time_span)
                    for v, l in zip(videos, labels)
            ]
    )
    pool.close()
    res = []
    for b in batches:
        for e in b:
            res.append(e)
    res = np.array(res)
    return res


def build_video_batch(v, l, batch_size, sequence_time_span):
    video = Video(v)
    label_df = pd.read_csv(l)
    video_frame_count = video.get_metadata().get('nframes')
    seq_nframes = int(video.frame_rate * sequence_time_span)
    batch_frame_count = int(batch_size * seq_nframes)
    batch_count = int(video_frame_count // batch_frame_count)
    nb = lambda batch_n: build_batch(
            v,
            batch_n,
            batch_frame_count,
            seq_nframes,
            video.frame_rate,
            label_df,
            batch_size,
            sequence_time_span
    )
    return [nb(i) for i in range(batch_count)]


def build_batch(
        video_name,
        batch_n,
        batch_frame_count,
        seq_nframes,
        fps,
        label_df,
        batch_size,
        sequence_time_span
):
    s = lambda seq_n: build_sequence(
            batch_n,
            seq_n,
            batch_frame_count,
            seq_nframes,
            fps,
            label_df,
            sequence_time_span
    )
    b = VideoBatch(video_name, tuple(s(j) for j in range(batch_size)))
    return b


def build_sequence(
        batch_n,
        seq_n,
        batch_frame_count,
        seq_nframes,
        fps,
        label_df,
        sequence_time_span
):
    # sequence in video
    start_seq = batch_n * batch_frame_count + seq_n * seq_nframes
    end_seq = start_seq + seq_nframes
    # label of the sequence
    t = start_seq // fps + sequence_time_span // 2
    dft = label_df[label_df.middle_frame_timestamp == t]
    seq_label = dft[dft.isin(ACTION_LABELS).action_id].empty
    if seq_label:
        seq_label = 1
    else:
        seq_label = 0
    seq = VideoSeq(start_seq, end_seq, seq_label)
    return seq
