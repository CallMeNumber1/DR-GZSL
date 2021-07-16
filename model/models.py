import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)


    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class encoder_template(nn.Module):

    def __init__(self,input_dim,latent_size,hidden_size_rule,device):
        super(encoder_template,self).__init__()



        if len(hidden_size_rule)==2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule)==3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1] , latent_size]

        modules = []
        for i in range(len(self.layer_sizes)-2):

            modules.append(nn.Linear(self.layer_sizes[i],self.layer_sizes[i+1]))
            modules.append(nn.ReLU())

        self.feature_encoder = nn.Sequential(*modules)

        self._mu_t = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._mu_p = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)

        self._logvar_t = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar_p = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)

        self.apply(weights_init)

        self.to(device)


    def forward(self,x):

        h = self.feature_encoder(x)


        mu_t =  self._mu_t(h)
        mu_p =  self._mu_p(h)
        logvar_t = self._logvar_t(h)
        logvar_p = self._logvar_p(h)

        return mu_t, logvar_t, mu_p, logvar_p

class decoder_template(nn.Module):

    def __init__(self,input_dim,output_dim,hidden_size_rule,device):
        super(decoder_template,self).__init__()
        '''
            TODO decoder 只有一个隐层？
        '''

        self.layer_sizes = [input_dim, hidden_size_rule[-1] , output_dim]

        self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),nn.Linear(self.layer_sizes[1],output_dim))

        self.apply(weights_init)

        self.to(device)
    def forward(self,x):

        return self.feature_decoder(x)
