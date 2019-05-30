#!/usr/bin/env python3
#########################################################################
# > File Name: init.py
# > Author: Samuel
# > Mail: enminghuang21119@gmail.com
# > Created Time: Thu May 30 21:30:23 2019
#########################################################################

import glob
import os
from main import INITIAL_EPSILON, save_obj, deque

def init_directory():
    try:
        os.mkdir("./objects")
    except:
        pass
    for files in glob.glob('./objects/*'):
        os.remove(files)

def init_cache():
    save_obj(INITIAL_EPSILON, "epsilon")
    save_obj(0, "time")
    save_obj(deque(), "Log")


if __name__ == '__main__':
    init_directory()
    init_cache()
