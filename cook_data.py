from asyncio import constants
from random import shuffle
import pandas as pd
import numpy as np
import os
from os import listdir, replace
from os.path import isfile, join, isdir
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import shutil

# define the features coming out of gt.txt
features = [
        "timestep",
        "obj_id",
        "shape",
        "visibility",
        "2d_bbox_x",
        "2d_bbox_y",
        "2d_bbox_w",
        "2d_bbox_h",
        "non-actor",
        "3d_pos_x",
        "3d_pos_y",
        "3d_pos_z",
        "3d_bbox_1_x",
        "3d_bbox_1_y",
        "3d_bbox_1_z",
        "3d_bbox_2_x",
        "3d_bbox_2_y",
        "3d_bbox_2_z",
        "3d_bbox_3_x",
        "3d_bbox_3_y",
        "3d_bbox_3_z",
        "3d_bbox_4_x",
        "3d_bbox_4_y",
        "3d_bbox_4_z",
        "3d_bbox_5_x",
        "3d_bbox_5_y",
        "3d_bbox_5_z",
        "3d_bbox_6_x",
        "3d_bbox_6_y",
        "3d_bbox_6_z",
        "3d_bbox_7_x",
        "3d_bbox_7_y",
        "3d_bbox_7_z",
        "3d_bbox_8_x",
        "3d_bbox_8_y",
        "3d_bbox_8_z",
        "revised_2d_bbox_x",
        "revised_2d_bbox_y",
        "revised_2d_bbox_w",
        "revised_2d_bbox_h"
    ]

#TODO: get 3d positions behind occluders

def get_gt(scene_name):
    return pd.read_csv(f"data/{scene_name}/gt.txt", header=None, names=features)


def get_dataset():
    master_df = pd.DataFrame([], columns=features + ['scene_name'])
    
    # randomly select 100 scenes
    scenes = np.array(listdir("data"))

    # go through each scene
    for scene_name in np.unique(scenes):
        if isdir(join("data", scene_name)):
            # get the ground truth tracks
            df = get_gt(scene_name)

            # get the unique object ids
            unique_objects = np.unique(df['obj_id'].to_numpy())
            df['scene_name'] = [scene_name for i in range(df.shape[0])]
            actors = []
            non_actors = []
            # filter out betweeen actors and non-actors
            for id in unique_objects:
                entry_idx = np.where(df['obj_id'].to_numpy() == id)
                if df.to_numpy()[entry_idx[0][0]][8] == 0:
                    actors.append(id)
                else:
                    non_actors.append(id)
            
            # save non-actor tracks to master list
            for id in non_actors:
                track_idx = np.where(df['obj_id'].to_numpy() == id)[0]
                master_df = pd.concat([master_df, df.iloc[track_idx]], axis=0)

    MAX_LEN = 300
    master_X = torch.zeros(size=(1,MAX_LEN,4))
    master_Y = []
    track_lengths = []
    saved_track = None
    saved_name = None
    scene_dict = {}
    shuffle(scenes)
    # for each scene name (shuffled)
    for s, scene_name in enumerate(scenes):
        # get all entries with that row
        idx = np.where(master_df['scene_name'] == scene_name)[0]
        scene_df = master_df.iloc[idx]
        objects = scene_df['obj_id'].unique()
        for obj_id in objects:
            # get the whole track
            track_idx = np.where(scene_df['obj_id'] == obj_id)
            track = scene_df[["3d_pos_x","3d_pos_y","3d_pos_z", "timestep"]].iloc[track_idx].to_numpy().astype(np.float64)

            # get the packed sequence of positions
            track = torch.tensor(track).unsqueeze(0)

            #TODO: Add more implausible paths (shunted or suddenly stopping objects)

            # if a half trajectory is saved
            if saved_track is not None:
                # stitch this prefix to the saved postfix
                false_track = torch.cat((track[:, :track.shape[1] // 2, :], saved_track), axis=1)

                # False flag
                master_Y.append(0)
                # remove the saved trajectory
                scene_dict[len(master_Y) - 1] = [scene_name, saved_name]

                # add the new false trajectory to the dataset w/ padding
                track_lengths.append([track.shape[1] // 2, saved_track.shape[1]])
                master_X[-1, :false_track.shape[1], :] = false_track
                master_X = torch.cat([master_X, torch.zeros(size=(1,MAX_LEN,4))], axis=0)
                saved_track = None
                saved_name = None

            # "True" flag 
            master_Y.append(1)
            scene_dict[len(master_Y) - 1] = [scene_name]
            
            # add the new true trajectory to the dataset
            track_lengths.append([track.shape[1]])
            master_X[-1, :track.shape[1], :] = track
            master_X = torch.cat([master_X, torch.zeros(size=(1,MAX_LEN,4))], axis=0)

            # save the postfix of this track
            saved_track = track[:, track.shape[1] // 2:, :]
            saved_name = scene_name

    master_X = master_X[:-1, :, :]
    master_Y = torch.FloatTensor(master_Y)

    return master_X, master_Y, track_lengths, scene_dict

class MCS_Sequence_Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
        # get all our data
        X, y, L, S = get_dataset()
        self.data = X[:, :, :-1]
        self.labels = y
        self.lengths = L
        self.scene_names = S
        self.timesteps = X[:, :, -1]

        self.max_length = X.shape[1]
    
    def __getitem__(self, index):
        row = self.data[index]
        packed_X = pack_padded_sequence(row.unsqueeze(1), [sum(self.lengths[index])], batch_first=False, enforce_sorted=False)
        timesteps = pack_padded_sequence(self.timesteps[index].unsqueeze(1), [sum(self.lengths[index])], batch_first=False, enforce_sorted=False)
        return packed_X.data, self.labels[index], timesteps.data, self.scene_names[index], self.lengths[index]

    def __len__(self):
        return len(self.data)

def export_videos(index, scene_names, tracklet_lengths, frame_timesteps):
    plausible = len(scene_names) == 1
    
    def funky_indexing(t, k, tracklet_lengths):
        return (t + (k * (tracklet_lengths[0]))).item()

    for k, scene in enumerate(scene_names):
        # get the paths to all the rgb images in the scene
        # print(os.listdir(f"data/{scene}/RGB/"))
        
        ft = torch.flatten(frame_timesteps)

        if not plausible:
            print(scene_names)
            print(tracklet_lengths)
            print(ft)

        tracklet = []
        if not plausible and k:
            tracklet = ft[int(tracklet_lengths[0]) + 1:]
        else:
            tracklet = ft[:int(tracklet_lengths[0])] # this might sometimes be the whole array, which is intended
        
        # given scene name and tracklet timesteps, get video
        # rgb_files = os.listdir(f"data/{scene[0]}/RGB/")
        for j, timestep in enumerate(torch.flatten(tracklet)):
            t = int(timestep)

            if not os.path.exists(f'failures/{"plausible" if plausible else "implausible"}/{index}'):
                os.mkdir(f'failures/{"plausible" if plausible else "implausible"}/{index}')

            # copy the file to our failures folder
            shutil.copy(
                f'data/{scene[0]}/RGB/{t:06n}.png', 
                f'failures/{"plausible" if plausible else "implausible"}/{index}/{funky_indexing(j, k, tracklet_lengths):06n}.png'
            )

    # convert it into a video
    os.system(f'ffmpeg -framerate 12 -i failures/{"plausible" if plausible else "implausible"}/{index}/%06d.png -pix_fmt yuv420p failures/{"plausible" if plausible else "implausible"}/{index}/output.mp4')