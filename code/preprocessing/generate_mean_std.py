
import sys
import json

import numpy as np
import os
import sys
from tqdm import tqdm


def generate_mean_std():
    stat_dict = {'mean': { }, 'std': { }}
    fo = '../../data/MIMIC/train_groundtruth'
    mean_dict = dict()
    std_dict = dict()
    for fi in tqdm(os.listdir(fo)):
        fi = os.path.join(fo, fi)
        if 'csv' in fi:
            feat_value_dict = dict()
            for iline, line in enumerate(open(fi)):
                if iline == 0:
                    columns = line.strip().split(',')
                else:
                    for ic, value in enumerate(line.strip().split(',')):
                        try:
                            value = float(value)
                            feat = columns[ic]
                            if feat not in feat_value_dict:
                                feat_value_dict[feat] = [value]
                            else:
                                feat_value_dict[feat].append(value)
                        except:
                            pass
            # print(feat_value_dict)
            for feat, vs in feat_value_dict.items():
                if len(vs) > 6:
                    if feat not in mean_dict:
                        mean_dict[feat] = []
                        std_dict[feat] = []
                    mean_dict[feat].append(np.mean(vs))
                    std_dict[feat].append(np.std(vs))
    for feat in mean_dict:
        stat_dict['mean'][feat] = np.mean(mean_dict[feat])
        stat_dict['std'][feat] = np.mean(std_dict[feat])
    with open('../../data/MIMIC/stat_dict.json', 'w') as f:
        f.write(json.dumps(stat_dict))



if __name__ == '__main__':
    generate_mean_std()
