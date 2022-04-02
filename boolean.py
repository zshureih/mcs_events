import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from bool_model import TransformerModel, generate_square_subsequent_mask
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from cook_data import MCS_Sequence_Dataset, export_videos


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
    criterion = nn.BCELoss()
    lr = 0.0001  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def eval(model, val_set, export_flag=False):
        model.eval()
        total_correct = 0
        outputs = []
        losses = []

        with torch.no_grad():
            i = 0
            for src, target, timesteps, scene_names, lengths in val_set:
                optimizer.zero_grad()
                src = src.permute(1, 0, 2).cuda()
                target = target.cuda()
                optimizer.zero_grad()

                src_mask = generate_square_subsequent_mask(src.size(0)).cuda()
                output = model(src, src_mask)

                loss = criterion(output.squeeze(0), target)
                losses.append(loss.item())

                outputs.append(output.item())
                if output < 0.5:
                    output = 0
                else:
                    output = 1

                if output == target:
                    total_correct += 1
                elif export_flag:
                    ## TODO: Export videos given scene names, lengths and timesteps
                    export_videos(i, scene_names, lengths, timesteps)
                
                i += 1

        return total_correct / len(val_set), losses

    def train(model, train_set):
        model.train()
        total_loss = 0
        log_interval = 200
        losses = []

        # inline method for now?
        i = 0
        for src, target, timesteps, scene_names, lengths in train_set:
            optimizer.zero_grad()
            src = src.permute(1, 0, 2).cuda()
            target = target.cuda()
            optimizer.zero_grad()

            src_mask = generate_square_subsequent_mask(src.size(0)).cuda()
            output = model(src, src_mask)
            loss = criterion(output.squeeze(0), target)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            i += 1

            if i % log_interval == 0 and i > 0:
                # lr = scheduler.get_last_lr()[0]
                cur_loss = total_loss / log_interval
                losses.append(cur_loss)
                print(f"epoch {epoch:3d} - batch {i:5d}/{len(train_set)} - loss={cur_loss} - lr = {lr}")
                total_loss = 0
        
        return losses

    best_val_acc = float('inf')
    avg_train_losses = []
    avg_val_losses = []
    val_accuracies = []
    num_epochs = 200
    for epoch in range(1, num_epochs + 1):
        #TODO: masking inputs for sequence prediction/generation
        train_losses = train(model, train_generator)
        if epoch == num_epochs:
            val_acc, val_losses = eval(model, test_generator, export_flag=True)
        else:
            val_acc, val_losses = eval(model, test_generator)

        # scheduler.step()
        
        avg_train_losses.append(np.mean(train_losses))
        avg_val_losses.append(np.mean(val_losses))
        val_accuracies.append(val_acc)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Average Loss per Epoch')
    ax1.plot(range(len(avg_train_losses)), avg_train_losses, label='Training Loss')
    ax1.plot(range(len(avg_val_losses)), avg_val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()

    ax2.set_title(f'Accuracy on Validation set ({len(test_generator)} samples)')
    ax2.plot(range(len(val_accuracies)), val_accuracies)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    plt.show(block=True)