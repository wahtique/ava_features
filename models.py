#!/usr/bin/env python
"""
Author : William Veal Phan
Creation date : 2019-10-29T11:43
"""

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam


def deep_lstm_model(image_shape, sequence_length, classes):
    (h, w, c) = image_shape
    sequence_shape = (sequence_length, h, w, c)
    video = Input(shape=sequence_shape)
    # cnn_base = VGG16(input_shape=image_shape,
    #                  weights="imagenet",
    #                  include_top=False)
    extractor_base = Xception(
            include_top=False,
            weights='imagenet',
            input_shape=image_shape,
            pooling='avg',
    )
    # cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    extractor = Model(extractor_base.input, extractor_base.output)
    extractor.trainable = False
    encoded_frames = TimeDistributed(extractor)(video)
    encoded_sequences = LSTM(64, return_sequences=True)(encoded_frames)
    encoded_sequence = LSTM(64)(encoded_sequences)
    hidden_layer = Dense(256, activation="relu")(encoded_sequence)
    outputs = Dense(classes, activation="softmax")(hidden_layer)
    model = Model(video, outputs)
    optimizer = Nadam(lr=0.002,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-08,
                      schedule_decay=0.004)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["categorical_accuracy"])
    return model
