#!/usr/bin/env python
# coding=utf-8


import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import traceback
import sys
import time
import numpy as np
from sklearn import metrics
import random
import json
from glob import glob
from collections import OrderedDict
from tqdm import tqdm


import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import data_loader
from models import tame
import myloss
import function
import parse, py_op

from flask import Flask
from flask import request
import json

sys.path.append('../tools')
app = Flask(__name__)

@app.route('/inference', methods=['GET', 'POST'])
def inference_route():
    pid = int(request.args.get('pid'))
    ctime = int(request.args.get('ctime'))
    infer = Inference()
    result = infer.inference(pid, ctime)
    return json.dumps(result)

@app.route('/uncertainty', methods=['POST'])
def uncertainty_route():
    request_data = request.get_json()
    pid = None
    ctime = None
    feat_list = None

    if request_data:
        if 'pid' in request_data:
            pid = int(request_data['pid'])
        if 'ctime' in request_data:
            ctime = int(request_data['ctime'])
        if 'feat_list' in request_data['feat_list']:
            feat_list = request_data['feat_list']
    
    infer = Inference()
    if feat_list:
        return infer.sample_uncertainty(pid, ctime,feat_list=feat_list)
    else:
        return infer.sample_uncertainty(pid, ctime)
        

args = parse.args
args.hidden_size = args.rnn_size = args.embed_size 
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0
if args.model != 'tame':
    args.use_ve = 0
    args.use_mm = 0
    args.use_ta = 0
if args.use_ve == 0:
    args.value_embedding = 'no'
print 'epochs,', args.epochs

def _cuda(tensor, is_tensor=True):
    if args.gpu:
        if is_tensor:
            return tensor.cuda(async=True)
        else:
            return tensor.cuda()
    else:
        return tensor

def get_lr(epoch):
    lr = args.lr
    return lr

    if epoch <= args.epochs * 0.5:
        lr = args.lr
    elif epoch <= args.epochs * 0.75:
        lr = 0.1 * args.lr
    elif epoch <= args.epochs * 0.9:
        lr = 0.01 * args.lr
    else:
        lr = 0.001 * args.lr
    return lr

def index_value(data):
    '''
    map data to index and value
    '''
    if args.use_ve == 0:
        data = Variable(_cuda(data)) # [bs, 250]
        return data
    data = data.numpy()
    index = data / (args.split_num + 1)
    value = data % (args.split_num + 1)
    index = Variable(_cuda(torch.from_numpy(index.astype(np.int64))))
    value = Variable(_cuda(torch.from_numpy(value.astype(np.int64))))
    return [index, value]

def train_eval(data_loader, net, loss, epoch, optimizer, best_metric, phase='train'):
    lr = get_lr(epoch)
    if phase == 'train':
        net.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        net.eval()

    loss_list, pred_list, label_list, mask_list = [], [], [], []
    feature_mm_dict = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_feature_mm_dict.json'))
    for b, data_list in enumerate(tqdm(data_loader)):
        data, label, mask, files = data_list[:4]
        data = index_value(data)
        if args.model == 'tame':
            pre_input, pre_time, post_input, post_time, dd_list = data_list [4:9]
            pre_input = index_value(pre_input)
            post_input = index_value(post_input)
            pre_time = Variable(_cuda(pre_time))
            post_time = Variable(_cuda(post_time))
            dd_list = Variable(_cuda(dd_list))
            neib = [pre_input, pre_time, post_input, post_time]

        label = Variable(_cuda(label)) # [bs, 1]
        mask = Variable(_cuda(mask)) # [bs, 1]
        if args.dataset in ['MIMIC'] and args.model == 'tame' and args.use_mm:
            output = net(data, neib=neib, dd=dd_list, mask=mask) # [bs, 1]
        elif args.model == 'tame' and args.use_ta:
            output = net(data, neib=neib, mask=mask) # [bs, 1]
        else:
            output = net(data, mask=mask) # [bs, 1]

        if phase == 'test':
            folder = os.path.join(args.result_dir, args.dataset, 'imputation_result')
            output_data = output.data.cpu().numpy()
            mask_data = mask.data.cpu().numpy().max(2)
            for (icu_data, icu_mask, icu_file) in zip(output_data, mask_data, files):
                icu_file = os.path.join(folder, icu_file.split('/')[-1].replace('.csv', '.npy'))
                np.save(icu_file, icu_data)
                if args.dataset == 'MIMIC':
                    with open(os.path.join(args.data_dir, args.dataset, 'train_groundtruth', icu_file.split('/')[-1].replace('.npy', '.csv'))) as f:
                        init_data = f.read().strip().split('\n')
                    # print(icu_file)
                    wf = open(icu_file.replace('.npy', '.csv'), 'w')
                    wf.write(init_data[0] + '\n')
                    item_list = init_data[0].strip().split(',')
                    if len(init_data) <= args.n_visit:
                        try:
                            assert len(init_data) == (icu_mask >= 0).sum() + 1
                        except:
                            pass
                            # print(len(init_data))
                            # print(sum(icu_mask >= 0))
                            # print(icu_file)
                    for init_line, out_line in zip(init_data[1:], icu_data):
                        init_line = init_line.strip().split(',')
                        new_line = [init_line[0]]
                        # assert len(init_line) == len(out_line) + 1
                        for item, iv, ov in zip(item_list[1:], init_line[1:], out_line):
                            if iv.strip() not in ['', 'NA']:
                                new_line.append('{:4.4f}'.format(float(iv.strip())))
                            else:
                                minv, maxv = feature_mm_dict[item]
                                ov = ov * (maxv - minv) + minv
                                new_line.append('{:4.4f}'.format(ov))
                        new_line = ','.join(new_line)
                        wf.write(new_line + '\n')
                    wf.close()


        loss_output = loss(output, label, mask)
        pred_list.append(output.data.cpu().numpy())
        loss_list.append(loss_output.data.cpu().numpy())
        label_list.append(label.data.cpu().numpy())
        mask_list.append(mask.data.cpu().numpy())

        if phase == 'train':
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()


    pred = np.concatenate(pred_list, 0)
    label = np.concatenate(label_list, 0)
    mask = np.concatenate(mask_list, 0)
    metric_list = function.compute_nRMSE(pred, label, mask)
    avg_loss = np.mean(loss_list)

    print('\nTrain Epoch %03d (lr %.5f)' % (epoch, lr))
    print('loss: {:3.4f} \t'.format(avg_loss))
    print('metric: {:s}'.format('\t'.join(['{:3.4f}'.format(m) for m in metric_list[:2]])))


    metric = metric_list[0]
    if phase == 'valid' and (best_metric[0] == 0 or best_metric[0] > metric):
        best_metric = [metric, epoch]
        function.save_model({'args': args, 'model': net, 'epoch':epoch, 'best_metric': best_metric})
    metric_list = metric_list[2:]
    name_list = args.name_list
    assert len(metric_list) == len(name_list) * 2
    s = args.model
    for i in range(len(metric_list)/2):
        name = name_list[i] + ''.join(['.' for _ in range(40 - len(name_list[i]))])
        print('{:s}{:3.4f}......{:3.4f}'.format(name, metric_list[2*i], metric_list[2*i+1]))
        s = s+ '  {:3.4f}'.format(metric_list[2*i])
    if phase != 'train':
        print('\t\t\t\t best epoch: {:d}     best MSE on missing value: {:3.4f} \t'.format(best_metric[1], best_metric[0])) 
        print(s)
    return best_metric

class Inference(object):
    def __init__(self):

        args.name_list = py_op.myreadjson(os.path.join(args.file_dir, args.dataset+'_feature_list.json'))[1:]
        args.output_size = len(args.name_list)
        test_files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_with_missing/*.csv')))
        if args.phase == 'test':
            train_phase, valid_phase, test_phase, train_shuffle = 'test', 'test', 'test', False
        else:
            train_phase, valid_phase, test_phase, train_shuffle = 'train', 'valid', 'test', True
        test_dataset = data_loader.DataBowl(args, test_files, phase=test_phase)
        # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        args.vocab_size = (args.output_size + 2) * (1 + args.split_num) + 5

        imputation_net = tame.AutoEncoder(args)
        prediction_net = tame.Classification(args)

    
        imputation_net = _cuda(imputation_net, 0)
        prediction_net = _cuda(prediction_net, 0)

        state_dict = torch.load('../../data/MIMIC/models/prediction.ckpt')
        prediction_net.load_state_dict(state_dict['state_dict'])

        # self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.imputation_net = imputation_net
        self.prediction_net = prediction_net



    def inference(self,id, ct):
        '''
        input:
            id: patient id
            ct: predict time
        output:
            risk:list
                time: ISO date
                risk: sepsis probability
        '''
        imputation_net = self.imputation_net
        prediction_net = self.prediction_net
        dataset = self.test_dataset

        data_list = list(dataset.get_input(id, ct))
        for i, x in enumerate(data_list):
            try:
                data_list[i] = torch.unsqueeze(x, 0)
            except:
                # traceback.print_exc()
                pass

        data, imputation_label, mask, files = data_list[:4]
        current_time = data_list[-2]
        data = index_value(data)
        sepsis_label = data_list[-1]
        mask = Variable(_cuda(mask)) # [bs, 1]

        imputation_output = imputation_net(data, mask=mask) # [bs, 1]
        prediction_output = prediction_net(data, mask=mask)

        risk_list = []
        for t, risk in zip(current_time.data.numpy()[0], prediction_output[0].cpu().data.numpy()):
            risk_list.append({
                'time': str(int(t)),
                'risk': str(risk[0])
                })
        print('Sample output')
        print(risk_list)
        return risk_list

    def sample_uncertainty(self, id, ct, feat_list=[]):
        '''
        input:
            id: patient id, range:[0 - 10]
            ct: predict time
            feat_list: a list of sample的lab test
        output:
            uncertainty: double
        '''
        imputation_net = self.imputation_net
        prediction_net = self.prediction_net
        dataset = self.test_dataset

        data_list = list(dataset.sample_input(id, ct, feat_list))

        data, mask, current_time = data_list
        data = index_value(data)
        mask = Variable(_cuda(mask)) # [bs, 1]

        prediction_output = prediction_net(data, mask=mask)

        uncertainty = prediction_output.cpu().data.numpy()[0].std()
        print('Sample output: Uncertainty')
        print(uncertainty)
        return uncertainty




if __name__ == '__main__':
    # infer = Inference()
    # id = 0
    # ctime = 11
    # infer.inference(id, ctime)
    # infer.sample_uncertainty(id, ctime)
    # infer.sample_uncertainty(id, ctime, ['wbc', 'gcs_min'])
    app.run()
