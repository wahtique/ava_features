#!/usr/bin/env python
"""
Some miscellaneous utility methods to make everything more readable.

Author : William Veal Phan
Creation date : 2019-10-17T14:52
"""
import os
from typing import List, Tuple

import wget
from tqdm import tqdm


# ======================== FILES PATH AND NAMES ========================

def explode_path(path: str) -> Tuple[str, str]:
    """
    Explode a path into its head and tail. In this context, path to dir and
    filename.

    Args:
        path: str path to a file

    Returns:
        dir, filename

    """
    return os.path.split(path)


def path_to_filename_no_ext(path: str) -> str:
    """
    Get the filename w/o extension from the path to file.

    Args:
        path: str, path to file

    Returns:
        filename without extension

    """
    fn = path_to_filename(path)
    return filename_to_filename_no_ext(fn)


def filename_to_filename_no_ext(filename: str) -> str:
    """
    Return the given filename w/o extension.

    Args:
        filename: str

    Returns:
        filename without extension

    """
    return str(filename.split(sep='.')[0])


def path_to_filename(path: str) -> str:
    """
    Get the filename  with extension from a path.

    Args:
        path: str, path to file

    Returns:
        filename

    """
    return os.path.split(path)[1]


def format_path(path: str) -> str:
    """
    Check if a given url or dir path ends with "/", and add this character if
    it does not.

    Args:
        path: str, path to directory

    Returns:
        same path with eventually "/" added at the end

    """
    if path[len(path) - 1] != '/':
        path += '/'
    return path


def read_list_of_files(path: str) -> List[str]:
    """
    Read a list of files from a file and return it as a list.

    Args:
        path: str path to a file containing file names, one line for each file

    Returns:
        a list of filenames as str

    """
    with open(path) as f:
        files = f.readline()
    return [x.strip() for x in files]


# ============================== DOWNLOAD ==============================

def download_file(file_name: str, out_dir: str, server_url: str) -> str:
    """
    Download a file from a url into a target directory. The directory MUST
    exist and the file must be available at {server_url}/{file_name}.
    Args:
        file_name: str
            name of the file to get from the server and to save in the target
            directory
        out_dir: str
            target directory in which to save the file
        server_url: str,
            url to the distant directory where the file can be found

    Returns:
        path to saved file

    """
    out_dir = format_path(out_dir)
    out_path = f'{out_dir}{file_name}'
    if not os.path.exists(out_path):
        server_url = format_path(server_url)
        f_url = f'{server_url}{file_name}'
        wget.download(f_url, out_path)
        return out_path


def download_files_in_list(
        file_list: List[str],
        out_dir: str,
        server_url: str
) -> List[str]:
    """
    Download files with file names in a list.

    Args:
        file_list: list of filenames
        out_dir: str, path to directory where the files will be written
        server_url: str, url to distant directory where the files can be found

    Returns:
        list of path to written files

    """
    written_files = []
    for file_name in tqdm(file_list):
        r = download_file(file_name, out_dir, server_url)
        written_files.append(r)
    return written_files


def download_from_list_file(
        list_in_file: str,
        out_dir: str,
        server_url: str
) -> List[str]:
    """
    Download files with names contained in a text file, one filename by line.

    Args:
        list_in_file: str,
            path to file containing the list of files to download
        out_dir: str, path to directory where the files will be written
        server_url: str, url to distant directory

    Returns:
        list of paths to written files
    """
    file_list = read_list_of_files(list_in_file)
    return download_files_in_list(file_list, out_dir, server_url)
