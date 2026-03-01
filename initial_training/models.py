import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader

class NCBF(nn.Module):
    def __init__(self,
                 seqInputHorizon, seqInputSize,      # seqInputSize will be 4 for (tilt, rel_x, rel_y, rel_z)
                 inputSize,                          # inputSize will be 2 for (ee_pos_x, ee_pos_y)
                 hiddens, seq_hiddens,               # val_input: 	(batch, 3)
                 CBF_gamma=0.9, activation="relu",   # seq_input:   (batch, Horizon, 4)
                 no_sequential=False,
                 regularizer_factor=None):
        super(NCBF, self).__init__()
        self.seqInputHorizon = seqInputHorizon
        self.seqInputSize = seqInputSize
        self.inputSize = inputSize
        self.CBF_gamma = CBF_gamma
        self.activation = activation
        self.regularizer_factor = regularizer_factor
        self.no_sequential = no_sequential
        
        # build model layers
        # sequential input processing
        if self.no_sequential:
            # if not using sequential processing
            self.seq_layers = nn.ModuleList()
            self.seq_layers.append(nn.Linear(seqInputSize, seq_hiddens[0]))
        else:
            # using LSTM to process sequential input
            self.lstm = nn.LSTM(seqInputSize, seq_hiddens[0], batch_first=True)
            # self.lstm = nn.LSTM(seqInputSize, seq_hiddens[0], num_layers=3, batch_first=True, dropout=0.2)
        
        # sequential post-processing layers
        self.seq_process_layers = nn.ModuleList()
        for i in range(len(seq_hiddens) - 1):
            self.seq_process_layers.append(nn.Linear(seq_hiddens[i], seq_hiddens[i+1]))
        
        # combined state and sequential feature processing layers
        self.combined_layers = nn.ModuleList()
        combined_input_size = inputSize + seq_hiddens[-1]
        for i, h in enumerate(hiddens):
            if i == 0:
                self.combined_layers.append(nn.Linear(combined_input_size, h))
            else:
                self.combined_layers.append(nn.Linear(hiddens[i-1], h))
        
        # output layer
        self.output_layer = nn.Linear(hiddens[-1] if hiddens else combined_input_size, 1)
        
        # activation function
        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "sin":
            self.act_fn = torch.sin
        else:
            self.act_fn = F.relu  # default using ReLU
            
        # optimizer will be set in fit method
        self.optimizer = None
    
    def forward(self, inputs):
        seq_input, val_input = inputs
        
        # process sequential input
        if self.no_sequential:
            x = seq_input
            for layer in self.seq_layers:
                x = self.act_fn(layer(x))
        else:
            # LSTM process sequential input
            x, _ = self.lstm(seq_input)
            # get the output of the last time step
            x = x[:, -1, :]
        
        # sequential post-processing
        for layer in self.seq_process_layers:
            x = self.act_fn(layer(x))
        
        # combine state input
        x = torch.cat([val_input, x], dim=1)
        
        # combined processing layers
        for layer in self.combined_layers:
            x = self.act_fn(layer(x))
        
        # output layer
        output = self.output_layer(x)
        
        return output
    
    def train_step(self, safe_batch, unsafe_batch, device, coeffs=None, 
                  derivative_bound=0.05, conservative_bound=0.01):
        safe_h_obj_xs, safe_h_obj_nxs, safe_ee_xs, safe_ee_nxs = [x.to(device) for x in safe_batch]
        unsafe_h_obj_xs, unsafe_ee_xs = [x.to(device) for x in unsafe_batch]

        self.train()
        self.optimizer.zero_grad()

        safe_vals = self([safe_h_obj_xs, safe_ee_xs])
        safe_nvals = self([safe_h_obj_nxs, safe_ee_nxs])
        unsafe_vals = self([unsafe_h_obj_xs, unsafe_ee_xs])

        pos_sign_loss = torch.mean(F.relu(safe_vals + conservative_bound))
        neg_sign_loss = torch.mean(F.relu(-unsafe_vals + conservative_bound))
        derivative_loss = torch.mean(F.relu(safe_nvals + (self.CBF_gamma - 1) * safe_vals + conservative_bound))
        bounding_loss = torch.mean(F.relu((safe_nvals - safe_vals - derivative_bound)))
        derivative_loss_med = 0

        if coeffs is None:
            loss = pos_sign_loss + neg_sign_loss + derivative_loss + bounding_loss + derivative_loss_med
        else:
            assert len(coeffs) == 4
            loss = (coeffs[0] * pos_sign_loss + coeffs[1] * neg_sign_loss + 
                    coeffs[2] * derivative_loss + coeffs[3] * bounding_loss + derivative_loss_med)

        loss.backward()
        self.optimizer.step()

        return (loss.item(), pos_sign_loss.item(), neg_sign_loss.item(), 
                derivative_loss.item(), bounding_loss.item(), derivative_loss_med)

    def fit(self, data, epoch=1000, verbose_num=10, lr=1e-3,
            coeffs=None, derivative_threshold=0.025, margin_threshold=0.01,
            save_iters=None, save_path=None, batch_size=256, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        safe_h_obj_xs, safe_h_obj_nxs, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_xs, unsafe_ee_xs = data

        safe_dataset = TensorDataset(
            torch.tensor(safe_h_obj_xs, dtype=torch.float32),
            torch.tensor(safe_h_obj_nxs, dtype=torch.float32),
            torch.tensor(safe_ee_xs, dtype=torch.float32),
            torch.tensor(safe_ee_nxs, dtype=torch.float32)
        )
        unsafe_dataset = TensorDataset(
            torch.tensor(unsafe_h_obj_xs, dtype=torch.float32),
            torch.tensor(unsafe_ee_xs, dtype=torch.float32)
        )
        safe_loader = DataLoader(safe_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        unsafe_loader = DataLoader(unsafe_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=lr, 
            weight_decay=self.regularizer_factor if self.regularizer_factor else 0
        )

        verbose_freq = max(1, int(epoch / verbose_num))

        for e in range(epoch):
            total_loss = 0
            num_batches = 0
            for safe_batch, unsafe_batch in zip(safe_loader, unsafe_loader):
                loss_tuple = self.train_step(
                    safe_batch, unsafe_batch, device,
                    coeffs=coeffs,
                    derivative_bound=derivative_threshold,
                    conservative_bound=margin_threshold
                )
                total_loss += loss_tuple[0]
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0

            if (e + 1) % verbose_freq == 0 or e == 0:
                print(f"Iter {e}: avg_loss={avg_loss}")

            if save_iters is not None and save_path is not None and (e + 1) % save_iters == 0:
                self.save(os.path.join(save_path, f"model_iter_{e+1}.pt"))
    
    def predict(self, states):
        """predict function, for model evaluation"""
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            if isinstance(states[0], np.ndarray):
                # if input is numpy array, convert to torch tensor
                seq_input = torch.tensor(states[0], dtype=torch.float32)
                val_input = torch.tensor(states[1], dtype=torch.float32)
                return self([seq_input, val_input]).cpu().numpy()
            else:
                # if already torch tensor
                return self(states).cpu().numpy()
    
    def save(self, path):
        """save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'seqInputHorizon': self.seqInputHorizon,
            'seqInputSize': self.seqInputSize,
            'inputSize': self.inputSize,
            'CBF_gamma': self.CBF_gamma,
            'activation': self.activation,
            'no_sequential': self.no_sequential,
            'regularizer_factor': self.regularizer_factor
        }, path)
    
    @classmethod
    def load(cls, path, hiddens, seq_hiddens):
        """load model"""
        checkpoint = torch.load(path)
        model = cls(
            seqInputHorizon=checkpoint['seqInputHorizon'],
            seqInputSize=checkpoint['seqInputSize'],
            inputSize=checkpoint['inputSize'],
            hiddens=hiddens, 
            seq_hiddens=seq_hiddens,
            CBF_gamma=checkpoint['CBF_gamma'],
            activation=checkpoint['activation'],
            no_sequential=checkpoint['no_sequential'],
            regularizer_factor=checkpoint['regularizer_factor']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class Normalizer:
    def __init__(self, input_size, batch_axis=0):
        self.input_size = input_size
        self.count = 0
        self.eps = 1e-2
        self.batch_axis = batch_axis
        self._mean = np.expand_dims(np.zeros(input_size, dtype=np.float32), batch_axis)
        self._var = np.expand_dims(np.ones(input_size, dtype=np.float32), batch_axis)
        self._cached_std_inverse = None
    
    def experience(self, x):
        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return
        self.count += count_x
        rate = x.dtype.type(count_x / self.count)
        mean_x = np.mean(x, axis=self.batch_axis, keepdims=True)
        var_x = np.var(x, axis=self.batch_axis, keepdims=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (
            var_x - self._var
            + delta_mean * (mean_x - self._mean)
        )
        self._cached_std_inverse = None
    
    def __call__(self, x, update=False):
        mean = np.broadcast_to(self._mean, x.shape)
        std_inv = np.broadcast_to(self._std_inverse, x.shape)
        if update:
            self.experience(x)
        normalized = (x - mean) * std_inv
        return normalized
    
    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5
        return self._cached_std_inverse
    
    def inverse(self, y):
        mean = np.broadcast_to(self._mean, y.shape)
        std = np.broadcast_to(np.sqrt(self._var + self.eps), y.shape)
        return y * std + mean
    
    def save_model(self, file_dir):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        np.save(os.path.join(file_dir, "mean.npy"), self._mean)
        np.save(os.path.join(file_dir, "var.npy"), self._var)
    
    def load_model(self, file_dir):
        self._mean = np.load(os.path.join(file_dir, "mean.npy"))
        self._var = np.load(os.path.join(file_dir, "var.npy"))