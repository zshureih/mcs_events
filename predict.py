import re
import pandas as pd
import numpy as np
import os
from os import listdir, replace
from os.path import isfile, join, isdir
import torch
from random import shuffle
from torch.utils.data import random_split
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data.dataset import Dataset
from sequence_model import TransformerModel, generate_square_subsequent_mask
import matplotlib.pyplot as plt

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
    track_lengths = []
    scene_dict = {}
    shuffle(scenes)
    # for each scene name (shuffled)
    for s, scene_name in enumerate(scenes):
        # get all entries with that row
        idx = np.where(master_df['scene_name'] == scene_name)[0]
        scene_df = master_df.iloc[idx]
        objects = scene_df['obj_id'].unique()
        for obj_id in objects:
            #TODO: Add more implausible paths (shunted or suddenly stopping objects)
            # get the whole track
            track_idx = np.where(scene_df['obj_id'] == obj_id)
            track = scene_df[["3d_pos_x","3d_pos_y","3d_pos_z", "timestep"]].iloc[track_idx].to_numpy().astype(np.float64)

            # get the packed sequence of positions
            track = torch.tensor(track).unsqueeze(0)

            # save the scene name 
            scene_dict[master_X.shape[0] - 1] = [scene_name]
            
            # add the new true trajectory to the dataset
            track_lengths.append([track.shape[1]])
            master_X[-1, :track.shape[1], :] = track
            master_X = torch.cat([master_X, torch.zeros(size=(1,MAX_LEN,4))], axis=0)

    master_X = master_X[:-1, :, :]

    return master_X, track_lengths, scene_dict

class MCS_Sequence_Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
        # get all our data
        X, L, S = get_dataset()
        self.data = torch.index_select(X[:, :, :-1], -1, torch.LongTensor([2, 1, 0]))
        self.lengths = L
        self.scene_names = S
        self.max_length = X.shape[1]

        self.timesteps = X[:, :, -1]
    
    def __getitem__(self, index):
        row = self.data[index]
        
        packed_X = pack_padded_sequence(row.unsqueeze(1), [sum(self.lengths[index])], batch_first=False, enforce_sorted=False)
        packed_X = packed_X.data
        # subtract the initial position of the trajectory from each position in trajectory to delta-ize
        packed_X = torch.sub(packed_X, packed_X[0, :].repeat((packed_X.size(0)),1))
        timesteps = pack_padded_sequence(self.timesteps[index].unsqueeze(1), [sum(self.lengths[index])], batch_first=False, enforce_sorted=False)
        
        return packed_X, timesteps.data.long(), self.scene_names[index], self.lengths[index]

    def __len__(self):
        return len(self.data)


def get_gt(scene_name):
    return pd.read_csv(f"data/{scene_name}/gt.txt", header=None, names=features)

def mask_input(src, timesteps):
    src = src.permute(1, 0, 2).cuda()
    target = src.detach().clone() # make a copy of our source and label it as the target

    # now let's mask the input
    # randomly select 15% of samples
    perm = torch.randperm(src.size(0), dtype=torch.int64)
    selected_idx = perm[:int(src.size(0) * 0.15)]

    # timesteps are shuffled, get 80/10/10 split
    num_selected = int(src.size(0) * 0.15)
    num_masked = int(num_selected * 0.8)
    num_randomized = num_masked + int(num_selected * 0.1)

    masked_idx = selected_idx[:num_masked].cuda()
    randomized_idx = selected_idx[num_masked:num_randomized]

    # mask idx 
    for t in masked_idx:
        src[t.item(), :, :] = torch.zeros((1, 3), dtype=torch.float64).cuda()

    for t in randomized_idx:
        src[t.item(), :, :] += torch.randn((1, 3), dtype=torch.float64).cuda()
    
    return src, target

if __name__ == "__main__":
     # Parameters
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 1}
    max_epochs = 100

    # grab the dataset
    full_dataset = MCS_Sequence_Dataset()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])

    train_generator = torch.utils.data.DataLoader(train_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)

    # okay let's start training
    model = TransformerModel(3, 128, 8, 128, 2, 0.2).cuda()
    criterion = nn.MSELoss()
    # lr = 0.00001  # learning rate
    lr = 0.01  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def eval(model, val_set):
        model.eval()
        losses = []
        outputs = []

        with torch.no_grad():
            for src, timesteps, scene_names, lengths in val_set:
                optimizer.zero_grad()
                src, target = mask_input(src, timesteps)

                optimizer.zero_grad()

                src_mask = generate_square_subsequent_mask(src.size(0)).cuda()
                output = model(src, src_mask)
                outputs.append([src, output, target, scene_names, lengths])

                loss = criterion(output.squeeze(0), target)
                losses.append(loss.item())

        return outputs[-1], losses

    def train(model, train_set):
        model.train()
        total_loss = 0
        log_interval = 200
        losses = []

        # inline method for now?
        i = 0
        for src, timesteps, scene_names, lengths in train_set:
            optimizer.zero_grad()

            # reshape and mask our src, create our target
            src, target = mask_input(src, timesteps)

            optimizer.zero_grad()

            src_mask = generate_square_subsequent_mask(src.size(0)).cuda()
            output = model(src, src_mask)
            loss = criterion(output.squeeze(0), target)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            i += 1

            if i % log_interval == 0 and i > 0:
                lr = scheduler.get_last_lr()[0]
                cur_loss = total_loss / log_interval
                losses.append(cur_loss)
                print(f"epoch {epoch:3d} - batch {i:5d}/{len(train_set)} - loss={cur_loss} - lr = {lr}")
                total_loss = 0
        
        return losses

    best_val_acc = float('inf')
    avg_train_losses = []
    avg_val_losses = []
    val_outputs = []
    for epoch in range(max_epochs):
        train_losses = train(model, train_generator)
        val_trajectories, val_losses = eval(model, test_generator)
        # scheduler.step()
        
        avg_train_losses.append(np.mean(train_losses))
        avg_val_losses.append(np.mean(val_losses))
        val_outputs.append(val_trajectories)

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(121)
    ax.set_title('Average Loss per Epoch')
    ax.plot(range(len(avg_train_losses)), avg_train_losses, label='Training Loss')
    ax.plot(range(len(avg_val_losses)), avg_val_losses, label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.legend()

    ax = fig.add_subplot(122, projection='3d')
    for info in val_outputs:
        src, output, target, name, length = info[0], info[1], info[2], info[3], info[4]
        ax.set_title(name)
        # print("name", name)
        # print("length", length)
        # print(output)

        x = src[:, :, 0].reshape(-1).cpu().numpy()
        y = src[:, :, 1].reshape(-1).cpu().numpy()
        z = src[:, :, 2].reshape(-1).cpu().numpy()
        ax.plot(x,y,z,c='red', label="source trajectory")

        x = target[:, :, 0].reshape(-1).cpu().numpy()
        y = target[:, :, 1].reshape(-1).cpu().numpy()
        z = target[:, :, 2].reshape(-1).cpu().numpy()
        ax.plot(x,y,z,c='blue',label="target trajectory")

        x = output[:, :, 0].reshape(-1).cpu().numpy()
        y = output[:, :, 1].reshape(-1).cpu().numpy()
        z = output[:, :, 2].reshape(-1).cpu().numpy()
        ax.plot(x,y,z,c='green',label="output trajectory")

        break

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()

    plt.show(block=True)