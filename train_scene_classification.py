import re
import pandas as pd
import numpy as np
import os
from os import listdir, replace
from os.path import isfile, join, isdir
from interaction_model import MultiObjectModel
import torch
from random import shuffle
from torch.utils.data import random_split
from torch import nn, Tensor
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
    
    scenes = np.array(listdir("cube_all_partial"))

    # go through each scene
    for scene_name in np.unique(scenes):
        if isdir(join("cube_all_partial", scene_name)):
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
            
            # save actor tracks to master list
            for id in actors:
                track_idx = np.where(df['obj_id'].to_numpy() == id)[0]
                master_df = pd.concat([master_df, df.iloc[track_idx]], axis=0)

    master_X = []
    track_lengths = []
    scene_dict = {}
    shuffle(scenes)
    # for each scene name (shuffled)
    for s, scene_name in enumerate(scenes):
        # get all entries with that row
        idx = np.where(master_df['scene_name'] == scene_name)[0]
        scene_df = master_df.iloc[idx]
        objects = scene_df['obj_id'].unique()
        # save the scene name 
        scene_dict[len(master_X)] = [scene_name]

        # get each object's track
        tracks = []
        track_length = []

        for obj_id in objects:
            # get the whole track
            track_idx = np.where(scene_df['obj_id'] == obj_id)
            track = scene_df[["3d_pos_x","3d_pos_y","3d_pos_z","timestep"]].iloc[track_idx].to_numpy().astype(np.float64)

            # get the packed sequence of positions
            track = torch.tensor(track).unsqueeze(0)    
            
            # add the new true trajectory to the dataset
            track_length.append(track.shape[1])
            
            # save the seqeunce without any padding
            tracks.append(track)

        master_X.append(tracks)
        track_lengths.append(track_length)

    return master_X, track_lengths, scene_dict

def find_gaps(timesteps):
    """Generate the gaps in the list of timesteps."""
    all_steps = []

    for i in range(int(timesteps[0]), int(timesteps[-1]) + 1):
        all_steps.append(i)

    for i in range(0, len(timesteps)):
        all_steps.remove(int(timesteps[i]))
    
    return all_steps

class MCS_Sequence_Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
        # get all our data
        X, L, S = get_dataset()

        self.data = {i: X[i] for i in range(len(X))}
        self.lengths = L
        self.scene_names = S
     
    def __getitem__(self, index):
        # get the scene
        scene = self.data[index]
        track_lengths = self.lengths[index]
        scene_name = self.scene_names[index]

        return scene, scene_name, track_lengths

    def __len__(self):
        return len(self.data)


def get_gt(scene_name):
    return pd.read_csv(f"cube_all_partial/{scene_name}/gt.txt", header=None, names=features)

def mask_input(src, timesteps, min_k=5):
    src = src.permute(1, 0, 2).cuda()
    target = src.detach().clone() # make a copy of our source and label it as the target

    # now let's mask the input
    # randomly select a k frame window to mask
    init_index = np.random.randint(0, src.size(0) - min_k)
    masked_idx = range(init_index, init_index + min_k)

    # mask idx 
    for t in masked_idx:
        src[t, :, :] = torch.full((1, 3), -99, dtype=torch.float64).cuda()

    return src, target, masked_idx

def get_deltas(source):
    # subtract the initial position of the trajectory from each position in trajectory to delta-ize
    comp_x = torch.cat((source[:, 0, :].unsqueeze(1), source[:, :-1, :]), axis=1)
    deltas = torch.sub(source, comp_x)
    return deltas.cuda()

def get_self_normalized(source):
    pass

def get_norm_from_deltas(src):
    # first position is always (0,0,0)
    # after that, values are added
    return torch.cumsum(src, axis=0)

def cook_tracks(object_tracks, track_lengths):
    # given N object tracks (x,y,z,t) from a single scene
    max_length = max(track_lengths)
    n = len(object_tracks)

    src = torch.full((n, max_length, 4), 99.0)

    for i in range(n):
        src[i, :track_lengths[i], :] = object_tracks[i]

    return src    

def get_relational_info(n):
    receiver_relations = np.zeros((1, n, n * (n - 1)), dtype=float)
    sender_relations   = np.zeros((1, n, n * (n - 1)), dtype=float)
    
    cnt = 0
    for i in range(n):
        for j in range(n):
            if(i != j):
                receiver_relations[:, i, cnt] = 1.0
                sender_relations[:, j, cnt]   = 1.0
                cnt += 1
    
    #There is no relation info in solar system task, just fill with zeros
    relation_info = np.zeros((1, n * (n - 1), 1))
    
    sender_relations   = torch.FloatTensor(sender_relations).cuda()
    receiver_relations = torch.FloatTensor(receiver_relations).cuda()
    relation_info      = torch.FloatTensor(relation_info).cuda()

    return sender_relations, receiver_relations, relation_info

if __name__ == "__main__":
    # Parameters
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 1}
    max_epochs = 10

    # grab the dataset
    full_dataset = MCS_Sequence_Dataset()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])

    train_generator = torch.utils.data.DataLoader(train_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)

    # okay let's start training
    model = MultiObjectModel(4, 128, 8, 128, 2, 0.2).cuda()
    
    # TODO: try huber loss
    criterion = nn.BCELoss()

    lr = 0.0001  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def eval(model, val_set, export_flag=False):
        model.eval()
        losses = []
        incorrect_scenes = []
        total_correct = 0

        with torch.no_grad():
            i = 0
            for object_tracks, scene_name, track_lengths in val_set:
                optimizer.zero_grad()

                n = len(track_lengths)
                plausibility = Tensor([1]) if "_plaus" in scene_name else Tensor([0])

                src = cook_tracks(object_tracks, track_lengths).cuda()

                output = model(src, track_lengths, get_relational_info(n))
                loss = criterion(output, plausibility.cuda())
                losses.append(loss.item())

                if output < 0.5:
                    output = 0
                else:
                    output = 1
                
                if output == plausibility.item():
                    total_correct += 1
                else:
                    incorrect_scenes.append(scene_name)
                
                i += 1

        return total_correct / len(val_set), losses


    def train(model, train_set):
        model.train()
        total_loss = 0
        log_interval = 10
        losses = []

        # online method for now?
        i = 0
        for object_tracks, scene_name, track_lengths in train_set:
            optimizer.zero_grad()

            n = len(track_lengths)
            plausibility = Tensor([1]) if "_plaus" in scene_name else Tensor([0])

            src = cook_tracks(object_tracks, track_lengths).cuda()

            output = model(src, track_lengths, get_relational_info(n))
            loss = criterion(output, plausibility.cuda())
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
        val_accuracies, val_losses = eval(model, test_generator)
        # scheduler.step()
        
        avg_train_losses.append(np.mean(train_losses))
        avg_val_losses.append(np.mean(val_losses))
        val_outputs.append(val_accuracies)

    # save model weights
    torch.save(model.state_dict(), f"{max_epochs}_interaction_model.pth")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Average Loss per Epoch')
    ax1.plot(range(len(avg_train_losses)), avg_train_losses, label='Training Loss')
    ax1.plot(range(len(avg_val_losses)), avg_val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()

    ax2.set_title(f'Accuracy on Validation set ({len(test_generator)} samples)')
    ax2.plot(range(len(val_outputs)), val_outputs)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    plt.show(block=True)