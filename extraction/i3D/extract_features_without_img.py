import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import io
import zipfile
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
import random

import numpy as np
from tqdm import tqdm
from collections import Counter
import math

from pytorch_i3d import InceptionI3d

import pdb
import json

def load_frame(frame_file, resize=False):

    data = Image.open(frame_file)

    if resize:
        data = data.resize((224, 224), Image.LANCZOS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data

def load_frame_dummy():

    data = np.zeros(shape=(224, 224, 3))

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data


def load_zipframe(zipdata, name, resize=False):

    stream = zipdata.read(name)
    data = Image.open(io.BytesIO(stream))

    assert(data.size[1] == 256)
    assert(data.size[0] == 340)

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data




def oversample_data(data): # (39, 16, 224, 224, 2)  # Check twice

    data_flip = np.array(data[:,:,:,::-1,:])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])   # ,:,16:240,58:282,:
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5,
        data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]




def load_rgb_batch(frames_dir, rgb_files, 
                   frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            if frame_indices[i][j] == -1:
                batch_data[i,j,:,:,:] = load_frame_dummy()
            else:
                batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, 
                    rgb_files[frame_indices[i][j]]), resize)

    return batch_data


def load_ziprgb_batch(rgb_zipdata, rgb_files, 
                   frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,:] = load_zipframe(rgb_zipdata, 
                rgb_files[frame_indices[i][j]], resize)

    return batch_data


def load_flow_batch(frames_dir, flow_x_files, flow_y_files, 
                    frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,0] = load_frame(os.path.join(frames_dir, 
                flow_x_files[frame_indices[i][j]]), resize)

            batch_data[i,j,:,:,1] = load_frame(os.path.join(frames_dir, 
                flow_y_files[frame_indices[i][j]]), resize)

    return batch_data


def load_zipflow_batch(flow_x_zipdata, flow_y_zipdata, 
                    flow_x_files, flow_y_files, 
                    frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,0] = load_zipframe(flow_x_zipdata, 
                flow_x_files[frame_indices[i][j]], resize)

            batch_data[i,j,:,:,1] = load_zipframe(flow_y_zipdata, 
                flow_y_files[frame_indices[i][j]], resize)

    return batch_data



def run(mode='rgb', load_model='', sample_mode='oversample', frequency=16,
    input_dir='', output_dir='', window_size = 3, batch_size=40, usezip=False, start=0, end=0, 
    i3d_type=None):
    if i3d_type == None:
        assert("No i3d type specified")
        
    chunk_size = 16

    assert(mode in ['rgb', 'flow'])
    assert(sample_mode in ['oversample', 'center_crop', 'resize'])
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    
    #i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224

        with torch.no_grad():
            b_data = b_data.cuda().float()
            b_features = i3d.extract_features(b_data)
        
        b_features = b_features.data.cpu().numpy()[:,:,0,0,0]
        return b_features

    video_names = sorted(os.listdir(input_dir))
    print("Video length is: ", len(video_names[start:end]))
    for video_name in video_names[start:end]:
        database = {}

        print(f"Processing: \n{video_name}")

        save_file = '{}.npy'.format(video_name)
        if save_file in os.listdir(output_dir):
            continue

        frames_dir = os.path.join(input_dir, video_name)
        frames_dir = f"{frames_dir}"

        if mode == 'rgb':
            if usezip:
                rgb_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'img.zip'), 'r')
                rgb_files = [i for i in rgb_zipdata.namelist() if i.startswith('img')]
            else:
                rgb_files = os.listdir(os.path.join(f"{frames_dir}"))

            rgb_files.sort()
            frame_cnt = len(rgb_files)

        else:
            if usezip:
                flow_x_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'flow_x.zip'), 'r')
                flow_x_files = [i for i in flow_x_zipdata.namelist() if i.startswith('x_')]

                flow_y_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'flow_y.zip'), 'r')
                flow_y_files = [i for i in flow_y_zipdata.namelist() if i.startswith('y_')]
            else:
                flow_x_files = os.listdir(os.path.join(f"{frames_dir}/flow_x"))
                flow_y_files = os.listdir(os.path.join(f"{frames_dir}/flow_y"))
                flow_x_files = list(map(lambda x: os.path.join(f"{frames_dir}/flow_x/{x}"), flow_x_files))
                flow_y_files = list(map(lambda x: os.path.join(f"{frames_dir}/flow_y/{x}"), flow_y_files))
                flow_x_frames = list(map(lambda x: load_frame(x, resize), flow_x_files))
                flow_y_frames = list(map(lambda x: load_frame(x, resize), flow_y_files))

            flow_x_files.sort()
            flow_y_files.sort()
            
            assert(len(flow_y_files) == len(flow_x_files))
            frame_cnt = len(flow_y_files)

        # Cut frames
        assert(frame_cnt > chunk_size)
        clipped_length = frame_cnt - chunk_size
        clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk

        frame_indices = [] # Frames to chunks
        for i in range(clipped_length // frequency):
            frame_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)])
        
        test_indices = [] # Frames to chunks
        for i in range(clipped_length // frequency):
            test_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)])
        
        window = window_size
        # window = np.random.randint(low=0, high=8)
        for i in range(len(frame_indices)):
            if i3d_type==1 or i3d_type==2:
                si = i
                while si == i:
                    si = random.randint(0,len(frame_indices) - 1)
                
                pick = random.randint(0,chunk_size - (window))
                place_dec = random.randint(0, 2)
                ## 0 is starting 1 is ending
                if place_dec:
                    place = chunk_size - (window)
                else:
                    place = 0
                if i3d_type == 1:
                    frame_indices[i][place:place+window] = test_indices[si][pick:pick+window]
                elif i3d_type == 2:
                    frame_indices[i][place:place+window] = [-1, -1, -1]

            database[i] = {
                "labels" : frame_indices[i],
                "window": window,
            }


        frame_indices = np.array(frame_indices)
        np.save(f"{REPRO_PATH}/{video_name}.npy", frame_indices)
        chunk_num = frame_indices.shape[0]

        batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        if sample_mode == 'oversample':
            full_features = [[] for i in range(10)]
        else:
            full_features = [[]]

        for batch_id in range(batch_num):
            require_resize = sample_mode == 'resize'
            if mode == 'rgb':
                if usezip:
                    batch_data = load_ziprgb_batch(rgb_zipdata, rgb_files, 
                        frame_indices[batch_id], require_resize)
                else:                
                    batch_data = load_rgb_batch(frames_dir, rgb_files, 
                        frame_indices[batch_id], require_resize)
            else:
                if usezip:
                    batch_data = load_zipflow_batch(
                        flow_x_zipdata, flow_y_zipdata,
                        flow_x_files, flow_y_files, 
                        frame_indices[batch_id], require_resize)
                else:
                    batch_data = load_flow_batch(frames_dir, 
                        flow_x_files, flow_y_files, 
                        frame_indices[batch_id], require_resize)

            if sample_mode == 'oversample':
                batch_data_ten_crop = oversample_data(batch_data)

                for i in range(10):
                    pdb.set_trace()
                    assert(batch_data_ten_crop[i].shape[-2]==224)
                    assert(batch_data_ten_crop[i].shape[-3]==224)
                    full_features[i].append(forward_batch(batch_data_ten_crop[i]))

            else:
                if sample_mode == 'center_crop':
                    batch_data = batch_data[:,:,16:240,58:282,:] # Centrer Crop  (39, 16, 224, 224, 2)
                
                assert(batch_data.shape[-2]==224)
                assert(batch_data.shape[-3]==224)
                full_features[0].append(forward_batch(batch_data))



        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)
        with open(f"{JSON_PATH}/{video_name}.json", "w") as f:
            json.dump(database, f)
        features = full_features.squeeze(0)
        features = np.swapaxes(features, 0, 1)
        np.save(os.path.join(output_dir, save_file), features)

        print('{} done: {} / {}, {}'.format(
            video_name, frame_cnt, clipped_length, features.shape))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--window_size', type=int)    
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--sample_mode', type=str)
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--usezip', dest='usezip', action='store_true')
    parser.add_argument('--no-usezip', dest='usezip', action='store_false')
    parser.add_argument('--seed', type = int, required = True)
    parser.add_argument('--epoch', type = int, required = True)
    parser.add_argument('--i3d_type', type = int, required = True)
    parser.set_defaults(usezip=True)

    args = parser.parse_args()
    output_dir = f"{args.output_dir}/i3d/{args.epoch}"
    REPRO_PATH = f"{args.output_dir}/index/{args.epoch}"
    JSON_PATH = f"{args.output_dir}/json/{args.epoch}"

    run(mode=args.mode, 
        load_model=args.load_model,
        sample_mode=args.sample_mode,
        input_dir=args.input_dir, 
        output_dir=output_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        frequency=args.frequency,
        usezip=args.usezip,
        start = args.start,
        end = args.end,
        i3d_type=args.i3d_type)
