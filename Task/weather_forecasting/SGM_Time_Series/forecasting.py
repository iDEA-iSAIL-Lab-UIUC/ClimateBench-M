import copy
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import time
from modules import *
from utils import *
from torch.optim import lr_scheduler
import yaml
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))
weather_forecasting_dir = os.path.dirname(current_dir)

data_dir = os.path.join(weather_forecasting_dir, 'processed_weather_data')
modules_dir = os.path.join(current_dir, 'saved_modules')

# Load Data
controlled_group_id_list = [0]
cross_validation_id_list = [1, 2, 3, 4]

for controlled_group_id in controlled_group_id_list:
    for cross_validation_id in cross_validation_id_list:

        X_train = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_train.pt')
        X_val = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_val.pt')
        X_test = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'X_test.pt')

        Y_test = torch.load(data_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'Y_test.pt')

        X_train = feature_l2_norm(X_train)
        X_val = feature_l2_norm(X_val)
        X_test = feature_l2_norm(X_test)

        num_counties = X_train.shape[0]
        window_size = 24

        X_train_RNN, Y_train_RNN = generate_sequential_data(torch.flatten(X_train, start_dim=1, end_dim=2).permute(1,0,2), period=window_size)  # period should be divided by 24
        num_training_samples = X_train_RNN.shape[0]  # num_training_samples = num_training_hours - 2, because every two period, we get a pair of (X, Y) sample

        X_val_RNN, Y_val_RNN = generate_sequential_data(torch.flatten(X_val, start_dim=1, end_dim=2).permute(1,0,2), period=window_size)
        num_val_samples = X_val_RNN.shape[0]

        X_test_RNN, Y_test_RNN = generate_sequential_data(torch.flatten(X_test, start_dim=1, end_dim=2).permute(1,0,2), period=window_size)
        num_test_samples = X_test_RNN.shape[0]

        DYNDAG = np.load(modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'best_ELBO_graph_seq.npy')  # (num_days, num_nodes, num_nodes)

        # fill in X_train_DYNDAG
        X_train_DYNDAG, X_val_DYNDAG, X_test_DYNDAG = get_training_val_test_DYNDAG(DYNDAG, num_training_samples, num_val_samples, num_test_samples, num_counties, window_size)

        batch_size = 8
        parallelization_mode = False  # parallel mode can allow larger batch_size

        train_combined = Combine_Dataset(X_train_RNN, Y_train_RNN, X_train_DYNDAG)
        training_dataloader_RNN = DataLoader(train_combined, batch_size=batch_size, shuffle=True)

        val_combined = Combine_Dataset(X_val_RNN, Y_val_RNN, X_val_DYNDAG)
        val_dataloader_RNN = DataLoader(val_combined, batch_size=batch_size, shuffle=True)

        num_epochs = 20
        max_grad_norm = 5
        base_lr = 0.01
        epsilon = 1e-3
        steps = [20, 30, 40, 50]
        lr_decay_ratio = 0.1

        yaml_file = os.path.join(current_dir, 'dcrnn.yaml')

        with open(yaml_file) as f:
            supervisor_config = yaml.safe_load(f)
        model_kwargs = supervisor_config.get('model')
        dcrnn_model = DCRNNModel(**model_kwargs).double()
        print(dcrnn_model)

        if parallelization_mode:
            dcrnn_model = nn.DataParallel(dcrnn_model)
        dcrnn_model = dcrnn_model.to(device)
        optimizer = torch.optim.Adam(dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lr_decay_ratio)

        batches_seen = 0
        best_dcrnn_model = copy.deepcopy(dcrnn_model)
        best_val_loss = np.inf

        best_epoch = 0
        for epoch in range(num_epochs):
            print('epoch: {}'.format(epoch))

            t = time.time()

            dcrnn_model.train()

            train_loss_list = []

            start_time = time.time()

            i = 0

            for x, y, dyn_adj in training_dataloader_RNN:
                optimizer.zero_grad()

                if parallelization_mode:
                    x = torch.flatten(x, start_dim=2)  # corresponding re-permute @ lines 272-273 modules.py
                    y = torch.flatten(y, start_dim=2)
                    x = x.to(device)
                    y = y.to(device)
                    dyn_adj = dyn_adj.to(device)
                else:
                    x = torch.flatten(x.permute(1,0,2,3), start_dim=2).to(device)  # after flatten (1, 0, 2*3) (batch_size, period, num_nodes*num_features)
                    y = torch.flatten(y.permute(1,0,2,3), start_dim=2).to(device)
                    dyn_adj = dyn_adj.to(device)  # (batch, num_nodes, num_nodes)

                output = dcrnn_model(x, dyn_adj, y, parallelization_mode, batches_seen)

                if parallelization_mode:
                    dim = output.shape[2]
                    loss = masked_mae_loss(y.permute(1, 0, 2), output.view(window_size, -1, dim))
                else:
                    loss = masked_mae_loss(y, output)

                train_loss_list.append(loss.item())

                loss.backward()
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(dcrnn_model.parameters(), max_grad_norm)

                optimizer.step()

            lr_scheduler.step()

            # evaluation on val
            val_loss_list = []

            print('\tvalidation starts')
            dcrnn_model.eval()

            for x, y, dyn_adj in val_dataloader_RNN:

                if parallelization_mode:
                    x = torch.flatten(x, start_dim=2)  # corresponding re-permute @ lines 272-273 modules.py
                    y = torch.flatten(y, start_dim=2)
                    x = x.to(device)
                    y = y.to(device)
                    dyn_adj = dyn_adj.to(device)
                else:
                    x = torch.flatten(x.permute(1,0,2,3), start_dim=2).to(device)  # after flatten (1, 0, 2*3) (batch_size, period, num_nodes*num_features)
                    y = torch.flatten(y.permute(1,0,2,3), start_dim=2).to(device)
                    dyn_adj = dyn_adj.to(device)  # (batch, num_nodes, num_nodes)

                output = dcrnn_model(x, dyn_adj, y, parallelization_mode, batches_seen)

                if parallelization_mode:
                    dim = output.shape[2]
                    loss = masked_mae_loss(y.permute(1, 0, 2), output.view(window_size, -1, dim))
                else:
                    loss = masked_mae_loss(y, output)

                val_loss_list.append(loss.item())

            val_loss_epoch = np.mean(val_loss_list)
            print('Epoch: {:04d}'.format(epoch),
                  'train mse loss: {:.10f}'.format(np.mean(train_loss_list)),
                  'val mse loss: {:.10f}'.format(val_loss_epoch),
                  'time: {:.4f}s'.format(time.time() - t))

            if val_loss_epoch <= best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss_epoch
                best_dcrnn_model = copy.deepcopy(dcrnn_model)

        print('Best epoch: {}, Best val loss: {}'. format(best_epoch, best_val_loss))
        torch.save(best_dcrnn_model, modules_dir + '/' + str(controlled_group_id) + '_' + str(cross_validation_id) + '_' + 'best_dcrnn_model.pt')

