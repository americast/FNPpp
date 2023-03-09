from data import *
from model import EncoderModules, DecoderModules, OutputModules
from util import *


import numpy as np
import torch
import torch.nn as nn
import time
import os
import pandas as pd
import random
import argparse
import json
import itertools
import matplotlib.pyplot as plt 
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

SEED = 17

WEEKS_AHEAD = 4
DAY_WEEK_MULTIPLIER = 7

class Hosp(nn.Module):
    def __init__(self, args: argparse.ArgumentParser):
        super(Hosp, self).__init__()

        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        self.get_hyperparameters()
        self.device = torch.device(args.dev)
        self.exp = args.exp
        self.pred_week = args.pred_week
        self.region = args.region
        self.start_week = EW_START_DATA
        min_sequence_length = 20

        start = time.time()

        ''' Process user input regarding regions '''
        global all_hhs_regions, regions
        if self.region=='all':
            regions = all_hhs_regions
        else:
            regions = self.region
        if type(regions) is str:
            regions = list([regions])

        ''' Import data for all regions '''
        self.initial_conditions = {}; self.end_point = {}
        r_seqs = []  # state sequences of features
        r_ys = []  # state targets
        for region in regions:
            X_state, y = get_state_train_data(region,self.pred_week)
            r_seqs.append(X_state.to_numpy())
            r_ys.append(y)

        r_seqs = np.array(r_seqs)  # shape: [regions, time, features]
        r_ys = np.array(r_ys)  # shape: [regions, time, 1]
        

        # Normalize
        # One scaler per state
        seq_scalers = dict(zip(regions, [StandardScaler() for _ in range(len(regions))]))
        ys_scalers = dict(zip(regions, [TorchStandardScaler() for _ in range(len(regions))]))
        r_seqs_norm, r_ys_norm = [], []
        for i, r in enumerate(regions):
            r_seqs_norm.append(seq_scalers[r].fit_transform(r_seqs[i],self.device))
            r_ys_norm.append(ys_scalers[r].fit_transform(r_ys[i],self.device))
        r_seqs_norm = np.array(r_seqs_norm)
        r_ys_norm = np.array(r_ys_norm)
        # two of them are used during training
        self.ys_scalers = ys_scalers

        states, seqs, seqs_masks, y, y_mask, time_seqs = [], [], [], [], [], []
        test_states, test_seqs, test_seqs_masks, test_time_seq = [], [], [], []
        for region, seq, ys in zip(regions, r_seqs_norm, r_ys_norm):
            # ys_weights = np.ones((ys.shape[0],1))
            # ys_weights[-14:] *= 5 
            seq, seq_mask, ys, ys_mask = create_window_seqs(seq,ys,min_sequence_length)
            # normal
            states.extend([region for _ in range(seq.shape[0])])
            seqs.append(seq)
            seqs_masks.append(seq_mask)
            y.append(ys)
            y_mask.append(ys_mask)
            # time sequences
            time_seq = create_time_seq(seq.shape[0],min_sequence_length+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER).unsqueeze(2)
            time_seqs.append(time_seq)
            # now fill up the test data
            test_states.append(region)
            test_seqs.append(seq[[-1]]); test_seqs_masks.append(seq_mask[[-1]])
            test_time_seq.append(time_seq[[-1]])

        # train and validation data, combine 
        regions_train = np.array(states, dtype="str").tolist()
        X_train = torch.cat(seqs,axis=0).float().numpy()
        X_mask_train = torch.cat(seqs_masks,axis=0).unsqueeze(2).float().numpy()
        y_train = torch.cat(y,axis=0).float().numpy()
        y_mask_train = torch.cat(y_mask,axis=0).float().numpy()
        time_train = torch.cat(time_seqs,axis=0).float().numpy()

        # same for test
        regions_test = np.array(test_states, dtype="str").tolist() 
        X_test = torch.cat(test_seqs,axis=0).float().numpy()
        X_mask_test = torch.cat(test_seqs_masks,axis=0).unsqueeze(2).float().numpy()
        time_test = torch.cat(test_time_seq,axis=0).float().numpy()
        # for scaling time module
        self.t_min = torch.tensor(time_train.min())
        # t_max is also useful for ode future 
        self.t_max = torch.tensor(time_train.max())

        # convert dataset to use in dataloader
        train_dataset = SeqData(regions_train, X_train, X_mask_train, y_train, y_mask_train, time_train)
        # note: y_val, y_mask_val, y_weights_val, rmse_val, are not needed at test time
        empty = np.zeros_like(regions_test)
        test_dataset = SeqData(regions_test, X_test, X_mask_test, empty, empty, time_test)
            
        # create dataloaders for each region
        self.data_loaders = {}
        for r in regions:
            idx = torch.tensor(np.isin(train_dataset.region,r)) #== r
            dset_train = torch.utils.data.dataset.Subset(train_dataset, np.where(idx)[0])
            r_train_loader = torch.utils.data.DataLoader(dset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True) 
            self.data_loaders[r] = r_train_loader
            # test data loader is small so we can use only one
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True) 

        print(time.time() - start ,' seconds')

        """ Create models, one for all region """
        # Current: one encoder for each region
        # Expect: a universal encoder for every region
        self.X_train = X_train
        self.encoder = EncoderModules(X_train.shape[2],self.device)
        self.decoder = DecoderModules(self.device)
        self.out_layer = OutputModules(device=self.device)


        self.losses = []
        self.data_losses = []
        self.train_start_flag = []
        self.epoch = 0



    def get_hyperparameters(self):
        """ using this for odenn and baselines"""
        model_params_path = './setup/'
        if not os.path.exists(model_params_path):
            os.makedirs(model_params_path)
        best_model_params_json_file = model_params_path+'Hosp-params.json'
        if os.path.exists(best_model_params_json_file):
            print('test mode, using existing params json')
            self.read_hyperparams_from_json(best_model_params_json_file) 
        else:
            raise Exception(f'no setup file {best_model_params_json_file}')

    def read_hyperparams_from_json(self, model_params_json_file):
        """ Reads hyperparameters for each, this are found in validation set"""
        with open(model_params_json_file) as f:
            self.model_metadata = json.load(f)
        
        self.num_epochs = self.model_metadata['NUM_EPOCHS']
        # self.keep_training = self.model_metadata['KEEP_TRAINING']
        self.lr = self.model_metadata['LEARNING_RATE']
        self.loss_weights = self.model_metadata['LOSS_WEIGHTS']
        self.batch_size = self.model_metadata['BATCH_SIZE']


    def forward_feature(self,region,X,X_mask,time_seq):
        ''' 
            Feature module forward pass
        '''  
        region_idx = {r: i for i, r in enumerate(regions)}
        def one_hot(idx, dim=len(region_idx)):
            ans = np.zeros(dim, dtype="float32")
            ans[idx] = 1.0
            return ans

        metadata = torch.tensor([one_hot(region_idx[r]) for r in region]).to(self.device)
            
        # X_embeds = self.encoder.mods.forward(X.transpose(1, 0), X_mask.transpose(1, 0))
        X_embeds = self.encoder.mods.forward_mask(X.transpose(1, 0), X_mask.transpose(1, 0), metadata)
        time_seq = time_seq[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,:]
        Hi_data = (time_seq - self.t_min)/(self.t_max - self.t_min)
        emb_prime = self.decoder.mods(Hi_data,X_embeds)
        states_prime = self.out_layer.mods(emb_prime) 
        return states_prime, emb_prime

    def compute_loss(self,states_prime,y,y_mask):
        """ 
            data loss for feature module 
            only loss needed for hosp
        """
        ys_data_mask = y_mask
        total_data_target_tokens = torch.sum(ys_data_mask != 0).cpu() # denominator of loss
        criterion = nn.MSELoss(reduction='none')
        data_loss = (
            criterion(
                states_prime[:,:,4],
                y[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,0]
            ) * ys_data_mask[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:]
            ).sum()
        data_loss /= total_data_target_tokens
        return data_loss


    def minibatch_train(self,optims,verbose=True):
        self.epoch += 1

        epoch_data_loss = [] 
        epoch_total_loss = []

        i = 0
        self.train()
        start_time = time.time()
        optims.zero_grad(set_to_none=True)
        for r in np.random.permutation(regions):  # one region at the time
            # backprop = False
            region, X, X_mask, y, y_mask, time_seq = next(iter(self.data_loaders[r])) 
            try:
                region_all = np.append(region_all,region)
                X_all = torch.cat((X_all, X.to(self.device, non_blocking=True)), dim=0)
                X_mask_all = torch.cat((X_mask_all, X_mask.to(self.device, non_blocking=True)), dim=0)
                time_seq_all = torch.cat((time_seq_all, time_seq.to(self.device, non_blocking=True)), dim=0)
                y_all = torch.cat((y_all, y.to(self.device, non_blocking=True)), dim=0)
                y_mask_all = torch.cat((y_mask_all, y_mask.to(self.device, non_blocking=True)), dim=0)
            except:
                region_all = region
                X_all = X.to(self.device, non_blocking=True)
                X_mask_all = X_mask.to(self.device, non_blocking=True)
                time_seq_all = time_seq.to(self.device, non_blocking=True)
                y_all = y.to(self.device, non_blocking=True)
                y_mask_all = y_mask.to(self.device, non_blocking=True)


            # ''' KD loss '''
            # if self.train_feat:
            # forward feature module 

        minibatch = np.random.choice(np.arange(len(region_all)), 512, replace=False)

        region_all = region_all[minibatch]
        X_all = X_all[minibatch,:,:]
        X_mask_all = X_mask_all[minibatch,:,:]
        time_seq_all = time_seq_all[minibatch,:,:]

        y_all = y_all[minibatch,:,:]
        y_mask_all = y_mask_all[minibatch,:]

        states_prime, emb_prime = self.forward_feature(region_all,X_all,X_mask_all,time_seq_all)
        data_loss = self.compute_loss(states_prime,y_all,y_mask_all)

        total_loss = self.loss_weights['feat_data']*data_loss

        total_loss.backward(retain_graph=False)

        optims.step()
        optims.zero_grad(set_to_none=True)
            
        epoch_total_loss.append(total_loss.detach().cpu().item())
        epoch_data_loss.append(data_loss.detach().cpu().item())
            
        epoch_data_loss = np.array(epoch_data_loss).mean()
        epoch_total_loss = np.array(epoch_total_loss).mean()

        elapsed = time.time() - start_time

        if verbose and self.epoch % 10 == 0:
            print('Epoch: %d, Data-F: %.2e, Time: %.3f'
                    %(self.epoch, epoch_data_loss.item(), elapsed))

        ''' save losses '''
        self.losses.append(epoch_total_loss)
        self.data_losses.append(epoch_data_loss)

    
    def evaluate(self):
        # evaluates in training
        total_mse_error = []
        with torch.no_grad():
            # go over region because architecture depends on it
            for r in np.random.permutation(regions): 
                region, X, X_mask, y, y_mask, time_seq = next(iter(self.data_loaders[r])) 
                # region = [region[0]] * X.size(1)
                self.eval()
                X = X.to(self.device, non_blocking=True)
                X_mask = X_mask.to(self.device, non_blocking=True)
                time_seq = time_seq.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y_mask = y_mask.to(self.device, non_blocking=True)

                # forward feature module 
                states_prime, _ = self.forward_feature(region,X,X_mask,time_seq)

                ys_data_mask = y_mask
                total_data_target_tokens = torch.sum(ys_data_mask != 0).cpu() # denominator of loss
                criterion = nn.MSELoss(reduction='none')
                mse_error = (
                    criterion(
                        states_prime[:,:,4],
                        y[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,0]
                    ) * ys_data_mask[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:]
                    ).sum()
                
                mse_error /= total_data_target_tokens
                total_mse_error.append(mse_error.cpu().item())

        rmse_error = np.sqrt(np.array(total_mse_error).mean())
        return rmse_error


    def _train(self,epochs):
        self.train_start_flag.append(self.epoch)  # save epoch where we start this training, used in loss plot

        ############## optimizers #############
        feat_params = itertools.chain(self.encoder.parameters(),self.decoder.parameters(),\
            self.out_layer.parameters())  
        optimizer_feat = torch.optim.Adam(feat_params, lr=self.lr, amsgrad=True)
        sch = optim.lr_scheduler.MultiStepLR(optimizer_feat, [1500,2000], gamma=0.1)
        # will search for 1% improvement at least
        self.epoch = 0
        self.train_start_flag.append(self.epoch)  # save epoch where we start this training, used in loss plot

        rmse_val = self.evaluate()
        print('initial validation rmse', rmse_val)
        for epoch in range(epochs):
            """ time solo """
            self.minibatch_train(optimizer_feat)
            sch.step()


    def _train_multi_traj(self, epochs, datasize=512):
        self.train_start_flag.append(self.epoch)  # save epoch where we start this training, used in loss plot

        ############## optimizers #############
        feat_params = itertools.chain(self.encoder.parameters(),self.decoder.parameters(),\
            self.out_layer.parameters())  
        optimizer_feat = torch.optim.Adam(feat_params, lr=self.lr, amsgrad=True)
        sch = optim.lr_scheduler.MultiStepLR(optimizer_feat, [1500,2000], gamma=0.1)
        # will search for 1% improvement at least
        self.epoch = 0
        self.train_start_flag.append(self.epoch)  # save epoch where we start this training, used in loss plot

        rmse_val = self.evaluate()
        print('initial validation rmse', rmse_val)
        for epoch in range(epochs):
            """ time solo """
            self.multi_traj_train(optimizer_feat, datasize=datasize)
            sch.step()

    
    def multi_traj_train(self, optims, datasize, verbose=True):
        self.epoch += 1

        epoch_data_loss = [] 
        epoch_total_loss = []

        self.train()
        start_time = time.time()
        optims.zero_grad(set_to_none=True)
        for r in np.random.permutation(regions):  # one region at the time
            # backprop = False
            region, X, X_mask, y, y_mask, time_seq = next(iter(self.data_loaders[r])) 
            try:
                region_all = np.append(region_all,region)
                X_all = torch.cat((X_all, X.to(self.device, non_blocking=True)), dim=0)
                X_mask_all = torch.cat((X_mask_all, X_mask.to(self.device, non_blocking=True)), dim=0)
                time_seq_all = torch.cat((time_seq_all, time_seq.to(self.device, non_blocking=True)), dim=0)
                y_all = torch.cat((y_all, y.to(self.device, non_blocking=True)), dim=0)
                y_mask_all = torch.cat((y_mask_all, y_mask.to(self.device, non_blocking=True)), dim=0)
            except:
                region_all = region
                X_all = X.to(self.device, non_blocking=True)
                X_mask_all = X_mask.to(self.device, non_blocking=True)
                time_seq_all = time_seq.to(self.device, non_blocking=True)
                y_all = y.to(self.device, non_blocking=True)
                y_mask_all = y_mask.to(self.device, non_blocking=True)


            # ''' KD loss '''
            # if self.train_feat:
            # forward feature module 
        if self.epoch == 1:
            self.minibatch = np.random.choice(np.arange(len(region_all)), datasize, replace=False)

        region_all = region_all[self.minibatch]
        X_all = X_all[self.minibatch,:,:]
        X_mask_all = X_mask_all[self.minibatch,:,:]
        time_seq_all = time_seq_all[self.minibatch,:,:]

        y_all = y_all[self.minibatch,:,:]
        y_mask_all = y_mask_all[self.minibatch,:]

        states_prime, emb_prime = self.forward_feature(region_all,X_all,X_mask_all,time_seq_all)
        data_loss = self.compute_loss(states_prime,y_all,y_mask_all)

        total_loss = self.loss_weights['feat_data']*data_loss

        total_loss.backward(retain_graph=False)

        optims.step()
        optims.zero_grad(set_to_none=True)
            
        epoch_total_loss.append(total_loss.detach().cpu().item())
        epoch_data_loss.append(data_loss.detach().cpu().item())
            
        epoch_data_loss = np.array(epoch_data_loss).mean()
        epoch_total_loss = np.array(epoch_total_loss).mean()

        elapsed = time.time() - start_time

        if verbose and self.epoch % 10 == 0:
            print('Epoch: %d, Data-F: %.2e, Time: %.3f'
                    %(self.epoch, epoch_data_loss.item(), elapsed))

        ''' save losses '''
        self.losses.append(epoch_total_loss)
        self.data_losses.append(epoch_data_loss)


    def predict_save(self,suffix=''):
        self.eval()
        with torch.no_grad():  #saves memory
            # only one batch
            region, X, X_mask, _, _, time_seq = next(iter(self.test_loader)) 
            self.eval()
            X = X.to(self.device, non_blocking=True)
            X_mask = X_mask.to(self.device, non_blocking=True)
            time_seq = time_seq.to(self.device, non_blocking=True)

            for i in range(len(region)):
                print('predict in ',region[i])
                states_prime, _ = self.forward_feature([region[i]],X[[i]],X_mask[[i]],time_seq[[i]])
                hosp_pred = states_prime[:,:,4].reshape(1,-1)
                hosp_pred = self.ys_scalers[region[i]].inverse_transform(hosp_pred)
                hosp_pred = hosp_pred.reshape(-1).detach().cpu().data.numpy()
                self.save_predictions(region[i],hosp_pred,''+suffix)


    def save_predictions(self, region, hosp_predictions, submodule):
        """
            Given an array w/ predictions, save as csv
        """
        data = np.array(
            [
                np.arange(len(hosp_predictions))+1,
                hosp_predictions
            ]
        )
        df = pd.DataFrame(data.transpose(),columns=['k_ahead','hospitalization'])
        df['k_ahead'] = df['k_ahead'].astype('int8')
        path = './results/{}/{}/'.format(self.exp, region)
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = 'hosp'+submodule
        file_name = 'preds_{}_{}_exp{}.csv'.format(model_name,self.pred_week,self.exp)
        df.to_csv(path+file_name,index=False)

    

    def plot_loss(self,suffix2=''):
        path = './figures/exp{}/'.format(self.exp)
        if not os.path.exists(path):
            os.makedirs(path)
        loss_names = ['total_loss','data_loss']
        losses = [self.losses, self.data_losses]
        
        time = [i for i in range(len(losses[0]))]  
        for loss_name, loss in zip(loss_names,losses):
            plt.yscale('log')
            plt.xlabel('epochs')
            plt.plot(time,loss,label=loss_name)
        if len(self.train_start_flag)>0:
            for flag in self.train_start_flag:
                plt.axvline(x = flag,color='y',linewidth=1,linestyle="dashed")
        plt.legend()
        i = 0
        if self.region=='all':
            figname = f'losses-{self.pred_week}-{self.exp}-{suffix2}-{i}.png'
        else:
            regions_str = '-'.join(self.region)
            figname = f'{regions_str}_losses-{self.pred_week}-{self.exp}-{suffix2}-{i}.png'
        while os.path.exists(path+figname):
            if self.region=='all':
                figname = f'losses-{self.pred_week}-{self.exp}-{suffix2}-{i}.png'
            else:
                regions_str = '-'.join(self.region)
                figname = f'{regions_str}_losses-{self.pred_week}-{self.exp}-{suffix2}-{i}.png'
            i += 1
        plt.savefig(path+figname)
        plt.close()


    def train_predict(self):

        # check if feat module is already there

        """  train feature params"""
        EPOCHS = self.num_epochs
        self._train(epochs=EPOCHS)

        """ save predictions """
        self.predict_save()
        # plot training losses
        self.plot_loss()


    def multi_train_predict(self, num_traj=50, datasize=256):
        EPOCHS = self.num_epochs

        for i in range(num_traj):
            self.encoder = EncoderModules(self.X_train.shape[2],self.device)
            self.decoder = DecoderModules(self.device)
            self.out_layer = OutputModules(device=self.device)

            self._train_multi_traj(epochs=EPOCHS, datasize=datasize)

            self.predict_save(suffix='traj{}'.format(i))
            self.plot_loss(suffix2='traj{}'.format(i))
        

    