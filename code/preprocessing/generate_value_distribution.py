#!/usr/bin/env python
# coding=utf-8


import sys

import os
import sys
import time
import numpy as np
from sklearn import metrics
import random
import json
from glob import glob
from collections import OrderedDict
from tqdm import tqdm


sys.path.append('../tools')
import parse, py_op
args = parse.args

def analyse_visit_length():
    files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_groundtruth/*')))
    n_list = []
    for ifi, fi in enumerate(tqdm(files)):
        if 'csv' not in fi:
            continue
        with open(fi) as f:
            s = f.read()
        n_list.append(len(s.split('\n')))
    print('Mean: ', np.mean(n_list), np.std(n_list))
    n_list = sorted(n_list)
    for i in range(0, len(n_list), int(len(n_list)/9)):
        print(n_list[i])

def analyse_totally_missing_rate():
    files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_groundtruth/*')))
    vis_list = []
    for ifi, fi in enumerate(tqdm(files)):
        if 'csv' not in fi:
            continue
        for i_line, line in enumerate(open(fi)):
            if i_line == 0:
                head = line.strip().split(',')
                vis = np.ones(len(head))
            else:
                data = line.strip().split(',')
                for i,v in enumerate(data):
                    if v!='NA' and len(v.strip()) > 0:
                        vis[i] = 0
        vis_list.append(vis)
    vis = np.array(vis_list).mean(0)
    assert len(vis) == len(head)
    for h,v in zip(head, vis):
        print(h, v)




def generate_feature_mm_dict():
    files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_groundtruth/*')))
    feature_value_dict = dict()
    for ifi, fi in enumerate(tqdm(files)):
        if 'csv' not in fi:
            continue
        for iline, line in enumerate(open(fi)):
            line = line.strip()
            if iline == 0:
                feat_list = line.split(',')
            else:
                data = line.split(',')
                for iv, v in enumerate(data):
                    if v in ['NA', '']:
                        continue
                    else:
                        feat = feat_list[iv]
                        if feat not in feature_value_dict:
                            feature_value_dict[feat] = []
                        feature_value_dict[feat].append(float(v))
    feature_mm_dict = dict()
    feature_ms_dict = dict()

    feature_range_dict = dict()
    for feat, vs in feature_value_dict.items():
        vs = sorted(vs)
        value_split = []
        for i in range(args.split_num):
            n = int(i * len(vs) / args.split_num)
            value_split.append(vs[n])
        value_split.append(vs[-1])
        feature_range_dict[feat] = value_split


        n = int(len(vs) / args.split_num)
        feature_mm_dict[feat] = [vs[n], vs[-n - 1]]
        feature_ms_dict[feat] = [np.mean(vs), np.std(vs)]

    py_op.mkdir(args.file_dir)
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_feature_mm_dict.json'), feature_mm_dict)
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_feature_ms_dict.json'), feature_ms_dict)
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_feature_list.json'), feat_list)
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_feature_value_dict_{:d}.json'.format(args.split_num)), feature_range_dict)

def split_data_to_ten_set():
    files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_with_missing/*')))
    np.random.shuffle(files)
    splits = []
    for i in range(10):
        st = int(len(files) * i / 10)
        en = int(len(files) * (i+1) / 10)
        splits.append(files[st:en])
    py_op.mywritejson(os.path.join(args.file_dir, args.dataset + '_splits.json'), splits)


def main():
    # generate_feature_mm_dict()
    # split_data_to_ten_set()
    # analyse_visit_length()
    analyse_totally_missing_rate()

if __name__ == '__main__':
    main()
