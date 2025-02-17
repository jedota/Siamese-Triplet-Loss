import os
import sys
import warnings
import shutil

import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from tensorflow.keras.applications import mobilenet_v2, resnet50, resnet
from tensorflow.keras.models import Model, load_model
from sklearn import metrics
from numpy.random import default_rng
from util import dataset_to_dict
import tensorflow_addons as tfa
import json
import socket
import augmentations
import callbacks as custom_callbacks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.experimental.AUTOTUNE

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def preprocess_images(images, labels, preprocesor):
    return preprocesor(images), labels


def load_image(filename, input_shape):
    label = tf.strings.to_number(filename[1], out_type=tf.dtypes.int64)
    filename = filename[0]

    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, input_shape[0:2])
    return image, label


def imgaug_augment(images):
    sequence = augmentations.build_augmentation_sequence_heavy()
    # self.augmentation_sequence.seed = self.rng.integers(0, 2 ** 16)
    images = tf.cast(images, tf.uint8).numpy()
    images = sequence.augment_images(images)
    return tf.cast(images, tf.float32)


def augment_images(images, seq_name='seq_color'):
    if seq_name == 'seq_color':
        ops = [[tf.image.random_contrast, dict(lower=0.3, upper=1.25)],
               [tf.image.random_hue, dict(max_delta=0.15)],
               [tf.image.random_saturation, dict(lower=0.25, upper=3)],
               [tf.image.adjust_gamma, dict(gamma=(2 * np.random.random() + 1) ** [-1, 1][np.random.randint(2)])]]
        ndx_op = np.random.randint(0, len(ops))
        images_aug = ops[ndx_op][0](**{**dict(image=images), **ops[ndx_op][1]})
    elif seq_name == 'imgaug_heavy':
        images_aug = tf.numpy_function(func=imgaug_augment, inp=[images], Tout=tf.float32)

    images_aug = tf.clip_by_value(images_aug, 0, 255)
    return images_aug


def train_siamese_network(**params):
    # Set seeds for repeatability
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    input_shape = params['input_shape']
    batch_size = params['batch_size']
    weights = params['weights']
    epochs = params['epochs']

    exp_dir = datetime.today().strftime('%Y%m%d_%H%M%S') + '_' + socket.gethostname()
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Read backbone_model network
    if params['backbone'] == 'mobilenetv2':
        alpha = params['backbone_params']['alpha']
        backbone_model = mobilenet_v2.MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha, weights=weights)
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = mobilenet_v2.preprocess_input
    elif params['backbone'] == 'resnet50':
        backbone_model = resnet50.ResNet50(include_top=True, input_shape=input_shape, weights=weights)
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = resnet50.preprocess_input
    elif params['backbone'] == 'resnet101':
        backbone_model = resnet.ResNet101(include_top=True, input_shape=input_shape, weights=weights)
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = resnet.preprocess_input
    elif params['backbone'] == 'resnet152':
        backbone_model = resnet.ResNet152(include_top=True, input_shape=input_shape, weights=weights)
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = resnet.preprocess_input
    elif params['backbone'] == 'custom':
        try:
            backbone_model = load_model(weights, compile=False)
        except Exception as e:
            sys.exit('If using a custom backbone, corresponding weights must be loaded.')
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = mobilenet_v2.preprocess_input

    # Add L2 normalization layer
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_norm')(backbone_model.output)
    model = Model(inputs=backbone_model.input, outputs=output)

    if 'unfreeze_from' in params.keys():
        if params['unfreeze_from'] is not None:
            for layer in model.layers:
                if layer.name == params['unfreeze_from']:
                    break
                else:
                    layer.trainable = False

    if 'path_train' in params.keys():
        dataset_train = dataset_to_dict(params['path_train'])
        n_samples_train = len(dataset_train)
        params['n_samples_train'] = n_samples_train

        dataset_train = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_train])
        dataset_train = dataset_train.shuffle(n_samples_train).map(lambda x: load_image(x, input_shape),
                                      num_parallel_calls=AUTOTUNE).batch(batch_size)

        if 'data_augmentation' in params.keys():
            if params['data_augmentation'] is not None:
                dataset_train = dataset_train.map(lambda x, y: (augment_images(x, params['data_augmentation']), y),
                                                  num_parallel_calls=AUTOTUNE)
        dataset_train = dataset_train.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    if 'path_val' in params.keys():
        dataset_val = dataset_to_dict(params['path_val'])
        n_samples_val = len(dataset_val)
        params['n_samples_val'] = n_samples_val
        dataset_val = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_val])
        dataset_val = dataset_val.map(lambda x: load_image(x, input_shape),
                                      num_parallel_calls=AUTOTUNE).batch(batch_size).map(
            lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    learning_rate = params['optimizer_params']['learning_rate']
    if params['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    elif params['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)

    if params['loss'] == 'contrastive_loss':
        loss_fn = tfa.losses.ContrastiveLoss()
    elif params['loss'] == 'triplet_semi_hard_loss':
        loss_fn = tfa.losses.TripletSemiHardLoss()
    if params['loss'] == 'triplet_hard_loss':
        loss_fn = tfa.losses.TripletHardLoss()

    model.compile(loss=loss_fn, optimizer=optimizer)
    model.summary()

    # Path to models (Best and latest)
    best_model_path = os.path.join(exp_dir, 'best_model')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    latest_model_path = os.path.join(exp_dir, 'latest_model')
    if not os.path.exists(latest_model_path):
        os.makedirs(latest_model_path)

    # Callbacks
    callbacks = list()
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=latest_model_path))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, save_best_only=True, monitor='val_loss'))

    callbacks.append(custom_callbacks.SaveTrainingData(latest_model_path, params))
    callbacks.append(custom_callbacks.SaveTrainingData(best_model_path, params, save_best_only=True, monitor='val_loss'))

    # Train the network
    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks)

def main():
    params = dict()

    params['path_train'] = ['path_file_to_train.txt']
    params['path_val'] = ['path_file_to_train.txt']
    params['loss'] = 'triplet_semi_hard_loss'
    params['backbone'] = 'mobilenetv2'
    params['backbone_params'] = {'alpha': 1.4}
    params['input_shape'] = (224, 224, 3)
    params['batch_size'] = 64
    params['epochs'] = 200
    params['use_rgb'] = True
    params['weights'] = 'imagenet'
    params['optimizer'] = 'adam'
    params['optimizer_params'] = {'learning_rate': 1e-05}
    params['data_augmentation'] = 'imgaug_heavy'
    
    train_siamese_network(**params)

if __name__ == '__main__':
    main()

