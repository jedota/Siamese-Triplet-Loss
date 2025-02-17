import copy

import numpy as np
import cv2
import os
import random
from tensorflow.keras.utils import to_categorical


def dataset_to_dict(input_data, shuffle_list=True, n_samples_per_class=None, remap_labels=None):
    file_ext = ['.jpg', '.JPG', '.png', '.PNG', '.jp2', '.JP2']
    dataset = list()

    if isinstance(input_data, list):
        for file in input_data:
            f = open(file, 'rt')
            for line in f:
                file_path, label = line.strip('\n').split(' ')
                sample_dict = dict()
                sample_dict['id'] = file_path
                sample_dict['label'] = int(label)
                dataset.append(sample_dict)
            f.close()
    elif os.path.isfile(input_data):
        f = open(input_data, 'rt')
        for line in f:
            file_path, label = line.strip('\n').split(' ')
            sample_dict = dict()
            sample_dict['id'] = file_path
            sample_dict['label'] = int(label)
            dataset.append(sample_dict)
        f.close()
    elif os.path.isdir(input_data):
        file_paths = list()
        for path, subdirs, files in os.walk(input_data):
            for name in files:
                if os.path.splitext(name)[1] in file_ext:
                    file_paths.append(os.path.join(path, name))
        base_dirs = list()
        for file_path in file_paths:
            base_dir = os.path.basename(os.path.split(file_path)[0])
            base_dirs.append(base_dir)
        base_dirs = sorted(list(set(base_dirs)))
        for file_path in file_paths:
            label = base_dirs.index(os.path.basename(os.path.split(file_path)[0]))
            sample_dict = dict()
            sample_dict['id'] = file_path
            sample_dict['label'] = int(label)
            dataset.append(sample_dict)

    if remap_labels is not None:
        for data in dataset:
            data['label'] = remap_labels[data['label']]

    if n_samples_per_class is not None:
        dataset_new = list()
        labels_set = sorted(list(set([x['label'] for x in dataset])))
        for lbl in labels_set:
            dataset_lbl = [x for x in dataset if x['label'] == lbl]
            # print(lbl, len(dataset_lbl))
            dataset_lbl = dataset_lbl[:n_samples_per_class]
            # print(lbl, len(dataset_lbl))
            dataset_new += dataset_lbl
        dataset = dataset_new

    if shuffle_list is True:
        random.shuffle(dataset)

    print(input_data)
    labels_set = sorted(list(set([x['label'] for x in dataset])))
    if len(labels_set) < 20:
        for lbl in labels_set:
            print('label: %d n_samples: %d' % (lbl, len([x for x in dataset if x['label'] == lbl])))
    print('Total classes in dataset: %d' % len(labels_set))
    print('Nmbr samples in dataset: %d' % len(dataset))

    return dataset


