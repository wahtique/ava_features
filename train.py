#!/usr/bin/env python
"""
Author : William Veal Phan
Creation date : 2019-10-14-15-47
"""
import logging as log
import multiprocessing as mp
import os
import time
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
    CSVLogger, TensorBoard

from AVAGenerator import AVAGenerator
from models import deep_lstm_model


def main():
    FPS = 5
    TIMESPAN = 2
    SEQ_LENGTH = FPS * TIMESPAN
    CHANNELS = 3  # rgb
    ROWS = 160
    COLS = 160
    CLASSES = 2  # interact / not interact
    BATCH_SIZE = 3
    SEQUENCE_SHAPE = (SEQ_LENGTH, ROWS, COLS, CHANNELS)  # hack for 5 fps
    IMAGE_SHAPE = (ROWS, COLS, CHANNELS)
    IMAGE_TARGET_SIZE = (ROWS, COLS)
    MODEL_NAME = 'xceptiondeeplstm'
    train_dir = '../data/AVA/dataset/AVA_action/train'
    val_dir = '../data/AVA/dataset/AVA_action/val'
    WORKERS = 8
    log.info(f"""
    {datetime.datetime.now()} : start 

    ======================= TRAIN PARAMETERS ==========================
    MODEL_NAME = {MODEL_NAME}
    TRAIN_DATA = {train_dir}
    VALIDATION_DATA = {val_dir}
    CLASSES = {CLASSES} 
    FRAMERATE = {FPS} fps
    TIMESPAN = {TIMESPAN} sc
    BATCH_SIZE = {BATCH_SIZE} sequences
    SEQUENCE_SHAPE = ({SEQ_LENGTH}, {ROWS}, {COLS}, {CHANNELS})
    # SEQ_LENGTH = {FPS * TIMESPAN} frames
    # CHANNELS = {CHANNELS}
    # HEIGHT = {ROWS} rows
    # WIDTH = {COLS} columns

    """)

    log.info("""
    ===================== DATA LOADING =============================== 
    """)
    start_train_gen = time.time()
    log.info(f'{datetime.datetime.now()} : Start loading training set')
    train_gen = AVAGenerator(
            dir_path=train_dir,
            batch_size=BATCH_SIZE,
            sequence_time_span=TIMESPAN,
            target_img_shape=IMAGE_TARGET_SIZE,
            target_fps=FPS,
            shuffle=True
    )
    middle_time = time.time()
    log.info(f'{datetime.datetime.now()} : End loading training set')
    log.info(f'{datetime.datetime.now()} : Start loading validation set')
    val_gen = AVAGenerator(
            dir_path=val_dir,
            batch_size=BATCH_SIZE,
            sequence_time_span=TIMESPAN,
            target_img_shape=IMAGE_TARGET_SIZE,
            target_fps=FPS,
            shuffle=True
    )
    end_time = time.time()
    log.info(f'{datetime.datetime.now()} : end loading validation set')
    log.info(f"""
    ======================= SUMMARY =============================

    Dataset init :
        - train : {middle_time - start_train_gen} sc
        - val : {end_time - middle_time} sc
    Steps per epoch for :
        - training : {len(train_gen)}
        - validation : {len(val_gen)}

    ====================== TRAINING ============================

    {datetime.datetime.now()} : configure TF
    """)

    gpus = tf.compat.v2.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = \
                tf.compat.v2.config.experimental.list_logical_devices(
                    'GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    log.info(f'{datetime.datetime.now()} : init model {MODEL_NAME}')
    m = deep_lstm_model(IMAGE_SHAPE, SEQ_LENGTH, 2)
    log.info(m.summary())
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
            filepath=os.path.join('data', 'checkpoints',
                                  f'{MODEL_NAME}-.{{epoch:}}-{{'
                                  f'val_loss:}}.hdf5'),
            verbose=1,
            save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('logs', MODEL_NAME))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(
            os.path.join('data', 'logs',
                         f'{MODEL_NAME}-training-{timestamp}.log'))

    log.info(f'{datetime.datetime.now()} : Start training')
    m.fit_generator(
            generator=train_gen,
            epochs=1,
            verbose=2,
            validation_data=val_gen,
            use_multiprocessing=True,
            max_queue_size=3 * WORKERS,
            workers=WORKERS,
            shuffle=False,  # custom shuffle already
            # callbacks=[checkpointer, tb, early_stopper]
    )


if __name__ == "__main__":
    log.basicConfig(level=log.DEBUG)
    mp.freeze_support()
    main()
