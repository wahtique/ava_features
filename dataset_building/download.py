#!/usr/bin/env python
"""
Downloading the Google AVA dataset from Python made easy.

Download url courtesy of CVDFoundation :
https://github.com/cvdfoundation/ava-dataset

Beware : the videos downloaded by this method will be the full movies used
in the AVA dataset. The train / val dataset for the action dataset weight
150Go !!!

Since AVA's labels only cover a portion of those movies, a second script
allowing to cut those videos to a more reasonable size is provided.

Author : William Veal Phan
Creation date : 2019-10-15
"""
from typing import List

import wget

from AVA_features.dataset_building.misc_utils import read_list_of_files, \
    download_from_list_file


# ============================== ACTION =============================

def download_ava_action_trainval_data(list_in_file: str, out_dir: str) -> \
        List[str]:
    ava_train_val_url = 'https://s3.amazonaws.com/ava-dataset/trainval/'
    return download_from_list_file(list_in_file, out_dir, ava_train_val_url)


def download_ava_action_trainval_filenames(out_path: str) -> str:
    u = 'https://s3.amazonaws.com/ava-dataset/annotations' \
        '/ava_file_names_trainval_v2.1.txt'
    return wget.download(u, out_path)


def download_ava_action_test_filenames(out_path: str) -> str:
    u = 'https://s3.amazonaws.com/ava-dataset/annotations' \
        '/ava_file_names_trainval_v2.1.txt'
    return wget.download(u, out_path)


def download_ava_action_labels(out_path: str) -> str:
    u = 'https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip'
    return wget.download(u, out_path)


def download_ava_action_test_data(list_in_file: str, out_dir: str) -> List[str]:
    ava_test_url = 'https://s3.amazonaws.com/ava-dataset/test/'
    return download_from_list_file(list_in_file, out_dir, ava_test_url)


# ============================== ACTIVE SPEAKER ==============================

def download_ava_activespeaker_trainval_data(list_in_file: str, out_dir: str) \
        -> List[str]:
    ava_as_trainval_url = 'https://s3.amazonaws.com/ava-dataset/trainval/'
    return download_from_list_file(list_in_file, out_dir, ava_as_trainval_url)


def download_ava_activespeaker_filenames(out_path: str) -> str:
    # same video files as speech
    u = 'https://s3.amazonaws.com/ava-dataset/annotations' \
        '/ava_speech_file_names_v1.txt'
    return wget.download(u, out_path)


def download_AVA_ActiveSpeaker_train_labels(out_path: str) -> str:
    u = 'https://s3.amazonaws.com/ava-dataset/annotations' \
        '/ava_activespeaker_train_v1.0.tar.bz2'
    return wget.download(u, out_path)


def download_AVA_ActiveSpeaker_val_labels(out_path: str) -> str:
    u = 'https://s3.amazonaws.com/ava-dataset/annotations' \
        '/ava_activespeaker_val_v1.0.tar.bz2'
    return wget.download(u, out_path)


# ============================== SPEECH ==============================

def download_ava_speech_trainval_data(list_in_file: str, out_dir: str) -> \
        List[str]:
    ava_speech_trainval_url = 'https://s3.amazonaws.com/ava-dataset/trainval/'
    return download_from_list_file(
            list_in_file,
            out_dir,
            ava_speech_trainval_url
    )


def download_ava_speech_labels(out_path: str) -> str:
    u = 'https://s3.amazonaws.com/ava-dataset/annotations' \
        '/ava_speech_labels_v1.csv'
    return wget.download(u, out_path)


def download_AVA_speech_filenames(out_path: str) -> str:
    # same files as ActiveSpeaker
    return download_ava_activespeaker_filenames(out_path)


# ============================= SCRIPT ================================


train_val_file = 'ava_file_names_trainval_v2.1.txt'
test_file = 'ava_file_names_test_v2.1.txt'
ava_train_val_files = read_list_of_files(train_val_file)
ava_test_files = read_list_of_files(test_file)

ava_test_dir = 'data/videos/AVA_action/test_vids/'

# os.chdir('AVA_action')
# os.chdir('train_val')
# for f in tqdm(ava_train_val_files):
#     if not os.path.exists(f):
#         url = f'{ava_train_val_url}{f}'
#         try:
#             wget.download(url, bar=None)
#         except:
#             continue
#
# os.chdir('../test')
#
# for f in tqdm(ava_test_files):
#     if not os.path.exists(f):
#         url = f'{ava_test_url}{f}'
#         try:
#             wget.download(url, bar=None)
#         except:
#             continue
