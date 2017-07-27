from __future__ import print_function

import argparse
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot
from keras import backend as K
import tensorflow as tf
import numpy as np
import cv2
from keras.backend.tensorflow_backend import set_session


def main(args):
    
    datagen = init(args)
    label = input_data(args.input_dir)
    
    for i in range(len(label)):
        files = os.listdir(os.path.join(args.input_dir, label[i]))
        num_files = len(files)
        img_arr = np.zeros((num_files, args.width, args.height, 3), dtype='uint8')
        
        for f in range(len(files)):
            img = cv2.imread(os.path.join(args.input_dir, label[i], files[f]))
            img = cv2.resize(img, (args.width, args.height))
            img_arr[f] = img
            
        if not len(img_arr) == 0:
            datagen.fit(img_arr)
            count = 0
            
            # output_dir
            if not os.path.exists(os.path.join(args.output_dir)):
                os.mkdir(args.output_dir)
                os.mkdir(os.path.join(args.output_dir, label[i]))
            
            for batch in datagen.flow(img_arr, batch_size=1, save_to_dir=os.path.join(args.output_dir, label[i]), save_prefix='data_aug', save_format='png'):
                count +=1
                if count > 50:
                    print ("Finished")
                    break
    

def input_data(input_dir):
    label=[]
    
    for a in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, a)):
            label.append(a)

    return label
    
def init(args):
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=args.gpu_memory_fraction
    set_session(tf.Session(config=config))
    
    print ("Set Tensorflow Session. GPU Memory Fraction = {}".format(args.gpu_memory_fraction))
    
    datagen = ImageDataGenerator(zca_whitening=args.zca_whitening,
                                 featurewise_std_normalization=args.featurewise_std_normalization,
                                 rotation_range=args.rotation_range,
                                 width_shift_range=args.width_shift_range,
                                 height_shift_range=args.height_shift_range,
                                 shear_range=args.shear_range,
                                 zoom_range=args.zoom_range,
                                 horizontal_flip=args.horizontal_flip,
                                 vertical_flip=args.vertical_flip,
                                 fill_mode=args.fill_mode
                                )
    return datagen


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--samplewise_center',
        default=True, 
        help="sample wise center."
    )
    parser.add_argument(
        '--featurewise_std_normalization',
        default=True,
        help="featurewise normalization"
    )
    parser.add_argument(
        '--samplewise_std_normalization',
        default=True,
        help="samplewise std normalization"
    )
    parser.add_argument(
        '--zca_whitening',
        default=True,
        help="zca whitening"
    )
    parser.add_argument(
        '--zca_epsilon',
        type=str,
        default='1e-6',
        help="zca epsilon"
    )
    parser.add_argument(
        '--rotation_range',
        type=float,
        default=40.,
        help='rotation range'
    )
    parser.add_argument(
        '--width_shift_range',
        type=float,
        default=0.1,
        help='width shift range'
    )
    parser.add_argument(
        '--height_shift_range',
        type=float,
        default=0.1,
        help='height shift range'
    )
    parser.add_argument(
        '--shear_range',
        type=float,
        default=0.1,
        help='shear range'
    )
    parser.add_argument(
        '--fill_mode',
        choices=['nearest', 'constant', 'reflect', 'wrap'],
        default='nearest',
        help='fill mode'
    )
    parser.add_argument(
        '--zoom_range',
        type=float,
        default=0.3,
        help='zoom range'
    )
    parser.add_argument(
        '--horizontal_flip',
        default=True,
        help='horizontal flip'
    )
    parser.add_argument(
        '--vertical_flip',
        default=True,
        help='vertical flip'
    )
    parser.add_argument(
        '--rescale',
        type=float,
        default=None,
        help='rescale'
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        default='',
        help='input directory'
    )
    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        default=0.5,
        help='gpu memory fraction'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        default='',
        help='output directory'
    )
    parser.add_argument(
        '--save_prefix',
        type=str,
        default='test',
        help='save prefix'
    )
    parser.add_argument(
        '--save_format',
        type=str,
        default='png',
        help='save format'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=50,
        help='width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=50,
        help='height'
    )
    
    args = parser.parse_args()
    
    main(args)
    