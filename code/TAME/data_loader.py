#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import json
import collections
import os
import random
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.append('../tools')
import py_op

def find_index(v, vs, i=0, j=-1):
    if j == -1:
        j = len(vs) - 1

    if v > vs[j]:
        return j + 1
    elif v < vs[i]:
        return i
    elif j - i == 1:
        return j

    k = int((i + j)/2)
    if v <= vs[k]:
        return find_index(v, vs, i, k)
    else:
        return find_index(v, vs, k, j)

def add_time_gap(idata, odata, n = 10):
    '''
    delete lines  with only meanbp_min, and urineoutput
    '''
    new_idata = []
    new_odata = []
    for iline,oline in zip(idata, odata):
        vs = []
        for v in oline.strip().split(','):
            if v in ['', 'NA']:
                vs.append(0)
            else:
                vs.append(1)
        vs[0] = 0
        vs[6] = 0
        vs[8] = 0
        if np.sum(vs) > 0:
            new_idata.append(iline)
            new_odata.append(oline)
    return new_idata, new_odata


class DataBowl(Dataset):
    def __init__(self, args, files, phase='train'):
        assert (phase == 'train' or phase == 'valid' or phase == 'test')
        self.args = args
        self.phase = phase
        self.files = files

        # self.sepsis_time = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_sepsis_time.json'))
        self.sepsis_time = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'hadm_sepsi_time_dict.json'))
        self.feature_mm_dict = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_feature_mm_dict.json'))
        self.feature_value_dict = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_feature_value_dict_{:d}.json'.format(args.split_num)))
        self.n_dd = 40
        if args.dataset in ['MIMIC']:
            self.ehr_list = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'ehr_list.json' ))
            self.ehr_id = { ehr: i+1 for i,ehr in enumerate(self.ehr_list) }
            self.args.n_ehr = len(self.ehr_id) + 1

    def map_input(self, value, feat_list, feat_index):

        # for each feature (index), there are 1 embedding vectors for NA, split_num=100 embedding vectors for different values
        index_start = (feat_index + 1)* (1 + self.args.split_num) + 1

        if value in ['NA', '']:
            if self.args.value_embedding == 'no':
                return 0
            return index_start - 1
        else:
            # print('""' + value + '""')
            value = float(value)
            if self.args.value_embedding == 'use_value':
                minv, maxv = self.feature_mm_dict[feat_list[feat_index]]
                v = (value - minv) / (maxv - minv + 10e-10)
                # print(v, minv, maxv)
                assert v >= 0
                # map the value to its embedding index
                v = int(self.args.split_num * v) + index_start
                return v
            elif self.args.value_embedding == 'use_order':
                vs = self.feature_value_dict[feat_list[feat_index]][1:-1]
                v = find_index(value, vs) + index_start
                return v
            elif self.args.value_embedding == 'no':
                minv, maxv = self.feature_mm_dict[feat_list[feat_index]]
                # v = (value - minv) / (maxv - minv)
                v = (value - minv) / maxv + 1
                v = int(v * self.args.split_num) / float(self.args.split_num)
                return v

    def map_output(self, value, feat_list, feat_index):
        if value in ['NA', '']:
            return 0
        else:
            value = float(value)
            minv, maxv = self.feature_mm_dict[feat_list[feat_index]]
            if maxv <= minv:
                print(feat_list[feat_index], minv, maxv)
            assert maxv > minv
            v = (value - minv) / (maxv - minv)
            # v = (value - minv) / (maxv - minv)
            v = max(0, min(v, 1))
            return v



    def get_mm_item(self, idx, prediction_time=-10000):
        input_file = self.files[idx]
        output_file = input_file.replace('with_missing', 'groundtruth')
        pid = input_file.split('/')[-1].split('.')[0]
        sepsis_time = self.sepsis_time.get(pid, 10000000)
        # sepsis_time = self.sepsis_time[pid]

        with open(output_file) as f:
            output_data = f.read().strip().split('\n')
        with open(input_file) as f:
            input_data = f.read().strip().split('\n')

        assert len(input_data) == len(output_data)
        mask_list, input_list, output_list, current_time  = [], [], [], []
        for iline in range(len(input_data)):
            inp = input_data[iline].strip()
            oup = output_data[iline].strip()

            if iline == 0:
                feat_list = inp.split(',')
            else:
                in_vs = inp.split(',')
                ou_vs = oup.split(',')
                ctime = int(inp.split(',')[0])
                if ctime > prediction_time and prediction_time > 0:
                    break
                mask, input, output = [], [], []
                rd = np.random.random(len(in_vs))
                for i, (iv, ov, ir, init_iv) in enumerate(zip(in_vs, ou_vs, rd, in_vs)):
                    if ir < 0.3 and self.phase=='train':
                        iv = 'NA'

                    if iv in ['NA', ''] and ov  not in ['NA', '']:
                        mask.append(1)
                    else:
                        mask.append(0)

                    input.append(self.map_input(iv, feat_list, i))
                    output.append(self.map_output(ov, feat_list, i))

                mask_list.append(mask)
                input_list.append(input)
                current_time.append(ctime)
                output_list.append(output)

        if prediction_time >0:
            pass
        elif len(mask_list) < self.args.n_visit:
            for _ in range(self.args.n_visit - len(mask_list)):
                # pad empty visit
                mask = [0 for _ in range(self.args.output_size + 1)]
                vs = [0 for _ in range(self.args.output_size + 1)]
                mask_list.append(mask)
                input_list.append(vs)
                current_time.append(current_time[-1])
                output_list.append(vs)
            # print(mask_list)
        else:
            mask_list = mask_list[- self.args.n_visit:]
            input_list = input_list[- self.args.n_visit:]
            current_time = current_time[- self.args.n_visit:]
            output_list = output_list[- self.args.n_visit:]


        # print(mask_list)
        mask_list = np.array(mask_list, dtype=np.int64)
        output_list = np.array(output_list, dtype=np.float32)
        if self.args.value_embedding == 'no' or self.args.use_ve == 0:
            input_list = np.array(input_list, dtype=np.float32)
        else:
            input_list = np.array(input_list, dtype=np.int64)

        input_list = input_list[:, 1:]
        output_list = output_list[:, 1:]
        mask_list = mask_list[:, 1:]

        sepsis_label = []
        for t in current_time:
            # print(type(sepsis_time))
            if sepsis_time - t <= 4:
                sepsis_label.append(1)
            else:
                sepsis_label.append(0)
        sepsis_label = np.array(sepsis_label, dtype=np.float32)
        current_time = np.array(current_time, dtype=np.float32)
        return torch.from_numpy(input_list), torch.from_numpy(output_list), torch.from_numpy(mask_list), \
                input_file, torch.from_numpy(current_time), torch.from_numpy(sepsis_label)




    def __getitem__(self, idx):
        return self.get_mm_item(idx)

    def get_input(self, idx, prediction_time):
        return self.get_mm_item(idx, prediction_time)


    def __len__(self):
        return len(self.files)

    def sample_idx(self, idx, sample_time, feat_list, mean_list):
        pass

    def __sample__(self, idx, sample_time, sample_dict):
        input_file = self.files[idx]
        output_file = input_file.replace('with_missing', 'groundtruth')
        pid = input_file.split('/')[-1].split('.')[0]
        sepsis_time = self.sepsis_time.get(pid, 10000000)
        # sepsis_time = self.sepsis_time[pid]

        with open(output_file) as f:
            output_data = f.read().strip().split('\n')
        with open(input_file) as f:
            input_data = f.read().strip().split('\n')

        assert len(input_data) == len(output_data)
        mask_list, input_list, output_list, current_time  = [], [], [], []
        time_list = [int(inp.split(',')[0]) for inp in input_data[1:]]
        sample_time = max([t for t in time_list if t <= sample_time])

        for iline in range(len(input_data)):
            inp = input_data[iline].strip()
            oup = output_data[iline].strip()

            if iline == 0:
                feat_list = inp.split(',')
            else:
                in_vs = inp.split(',')
                ou_vs = oup.split(',')
                ctime = int(inp.split(',')[0])
                mask, input, output = [], [], []
                rd = np.random.random(len(in_vs))
                for i, (iv, ov, ir, init_iv) in enumerate(zip(in_vs, ou_vs, rd, in_vs)):
                    if ir < 0.3 and self.phase=='train':
                        iv = 'NA'

                    if ctime == sample_time:
                        feat = feat_list[i]
                        if feat in sample_dict:
                            iv = str(sample_dict[feat])
                            ov = iv

                    if iv in ['NA', ''] and ov  not in ['NA', '']:
                        mask.append(1)
                    else:
                        mask.append(0)

                    input.append(self.map_input(iv, feat_list, i))
                    output.append(self.map_output(ov, feat_list, i))

                mask_list.append(mask)
                input_list.append(input)
                current_time.append(ctime)
                output_list.append(output)

        if len(mask_list) < self.args.n_visit:
            for _ in range(self.args.n_visit - len(mask_list)):
                # pad empty visit
                mask = [0 for _ in range(self.args.output_size + 1)]
                vs = [0 for _ in range(self.args.output_size + 1)]
                mask_list.append(mask)
                input_list.append(vs)
                # current_time.append(current_time[-1])
                current_time.append(100000)
                output_list.append(vs)
            # print(mask_list)
        else:
            mask_list = mask_list[- self.args.n_visit:]
            input_list = input_list[- self.args.n_visit:]
            current_time = current_time[- self.args.n_visit:]
            output_list = output_list[- self.args.n_visit:]


        # print(mask_list)
        mask_list = np.array(mask_list, dtype=np.int64)
        output_list = np.array(output_list, dtype=np.float32)
        if self.args.value_embedding == 'no' or self.args.use_ve == 0:
            input_list = np.array(input_list, dtype=np.float32)
        else:
            input_list = np.array(input_list, dtype=np.int64)

        input_list = input_list[:, 1:]
        output_list = output_list[:, 1:]
        mask_list = mask_list[:, 1:]

        sepsis_label = []
        for t in current_time:
            # print(type(sepsis_time))
            if sepsis_time - t <= 4:
                sepsis_label.append(1)
            else:
                sepsis_label.append(0)
        sepsis_label = np.array(sepsis_label, dtype=np.float32)
        return torch.from_numpy(input_list), torch.from_numpy(output_list), torch.from_numpy(mask_list), \
                input_file,torch.from_numpy(sepsis_label)


    def sample_input(self, idx, sample_time, sample_feat=[]):
        stat_dict = py_op.myreadjson(os.path.join(self.args.data_dir, self.args.dataset, 'stat_dict.json'))
        mean_dict = stat_dict['mean']
        std_dict = stat_dict['std']

        input_file = self.files[idx]
        with open(input_file) as f:
            input_data = f.read().strip().split('\n')
        sample_line = ''
        feat_list = input_data[0].strip().split()
        last_observed_value = { }

        # print(sample_time)
        for line in input_data[1:]:
            # print(line.split(',')[0] )
            if int(line.split(',')[0]) <= sample_time:
                sample_line = line
                for i, v in enumerate(line.strip().split()):
                    try:
                        last_observed_value[feat_list[i]] = float(v)
                    except:
                        pass
            else:
                break
        assert len(sample_line) > 0
        sample_feat_list = []

        for feat, value in zip(input_data[0].strip().split(','), sample_line.strip().split(',')):
            if feat not in sample_feat and len(sample_feat):
                continue
            try:
                value = float(value)
            except:
                sample_feat_list.append(feat)
        # print(sample_feat_list)


        sample_value_list = []
        sample_mask_list = []
        for feat in sample_feat_list:
            noise = np.random.normal(loc=0, scale=1, size=100)
            mean = last_observed_value.get(feat, mean_dict[feat])
            value = mean + std_dict[feat] * noise
            sample_value_list.append(value)

        sample_list = []
        mask_list = []
        for i in range(len(sample_value_list[0])):
            sample_dict = dict()
            for feat, value in zip(sample_feat_list, sample_value_list):
                sample_dict[feat] = value[i]
            data = list(self.__sample__(idx, sample_time, sample_dict))
            sample_list.append(data[0])
            mask_list.append(data[2])

        input = torch.stack(sample_list, 0)
        mask = torch.stack(mask_list, 0)
        return input, mask, data[-2]








