#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 01:07:39 2019

@author: roope
"""

import pickle
import os

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                'pickled_objects', 
                'classifier.pkl'), 'rb'))