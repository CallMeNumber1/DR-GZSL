#vaemodel
import copy
from math import hypot
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
import final_classifier as  classifier
import models

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Model(nn.Module):

    def __init__(self,hyperparameters):
        super(Model,self).__init__()
        self.shuffle_factor=hyperparameters['shuffle_factor']
        self.sc_lr = hyperparameters['sc_lr']
        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources  = ['resnet_features',self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots'] # 0
        self.latent_size = hyperparameters['latent_size'] # 64
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][0] # 200
        self.att_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][1] # 0
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2] # 400
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3] # 0
        self.reco_loss_function = hyperparameters['loss'] # l1
        self.nepoch = hyperparameters['epochs'] # 100
        self.lr_cls = hyperparameters['lr_cls'] # 0.001
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction'] # Ture
        # self.cls_train_epochs = hyperparameters['cls_train_steps'] # 23
        self.cls_train_epochs = 60 # 23
        '''
            此处的dataloader是自己写的，而不是utils.data.dataloader，self.DATASET='CUB', auxiliary_data_source='attribute'
        '''
        self.dataset = dataloader( self.DATASET, copy.deepcopy(self.auxiliary_data_source) , device= self.device )

        if self.DATASET=='CUB':
            self.num_classes=200
            self.num_novel_classes = 50
        elif self.DATASET=='SUN':
            self.num_classes=717
            self.num_novel_classes = 72
        elif self.DATASET=='AWA1' or self.DATASET=='AWA2':
            self.num_classes=50
            self.num_novel_classes = 10

        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        # Here, the encoders and decoders for all modalities are created and put into dict
        '''
            datatype: ['resnet_features', 'attribute']
            feature_dimensions: [2048, 312]
        '''
        self.encoder = {}
        
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):

            self.encoder[datatype] = models.encoder_template(dim,self.latent_size,self.hidden_size_rule[datatype],self.device)

            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.hidden_size_rule[datatype],self.device)

        '''
            两个模态各加一个辅助分类器，以计算shuffle classification loss
        '''
        self.aux_cla = {}
        for datatype in self.all_data_sources:
            self.aux_cla[datatype] = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes).to(self.device)
            # self.aux_cla[datatype] = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes-self.num_novel_classes)
            self.aux_cla[datatype].apply(models.weights_init)
        self.shuffle_classification_criterion = nn.NLLLoss()
        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters()) # []
        for datatype in self.all_data_sources:
            parameters_to_optimize +=  list(self.encoder[datatype].parameters())
            parameters_to_optimize +=  list(self.decoder[datatype].parameters())
            # 辅助分类器加到要优化的参数列表中
            # parameters_to_optimize += list(self.aux_cla[datatype].parameters())

        '''
            TODO 辅助分类器和解耦VAE的其他模块设置不同学习率
        '''
        parameters_cls = []
        for datatype in self.all_data_sources:
            parameters_cls += list(self.aux_cla[datatype].parameters())
        self.optimizer  = optim.Adam( parameters_to_optimize ,lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        self.optimizer  = optim.Adam( [{'params':parameters_to_optimize ,'lr':hyperparameters['lr_gen_model']},
                                    {'params':parameters_cls, 'lr':self.sc_lr}], 
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label

    def trainstep(self, img, att, label):
        '''
            TODO 调整辅助分类器的学习率，以查看进一步降低分类损失的效果
        '''
        # self.sc_lr *= (0.1 ** (self.current_epoch // 50))
        # self.optimizer.param_groups[1]['lr'] = self.sc_lr
        
        ##############################################
        # Encode image features and additional
        # features
        ##############################################

        # mu_img, logvar_img = self.encoder['resnet_features'](img)
        mu_t_img, logvar_t_img, mu_p_img, logvar_p_img = self.encoder['resnet_features'](img)
        # z_from_img = self.reparameterize(mu_img, logvar_img)
        z_t_from_img = self.reparameterize(mu_t_img, logvar_t_img)
        z_p_from_img = self.reparameterize(mu_p_img, logvar_p_img) # torch.Size([50(b), 64])
        z_from_img = z_t_from_img+z_p_from_img
        '''
            将类别无关特征进行shuffle，然后再和之前的类别相关特征组合，送入分类器
        '''
        perm = torch.randperm(z_p_from_img.size(0))
        shuffled_z_p_from_img = z_p_from_img[perm]
        shuffled_z_from_img = shuffled_z_p_from_img + z_t_from_img
        # mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        mu_t_att, logvar_t_att, mu_p_att, logvar_p_att = self.encoder[self.auxiliary_data_source](att)
        # z_from_att = self.reparameterize(mu_att, logvar_att)
        z_t_from_att = self.reparameterize(mu_t_att, logvar_t_att)
        z_p_from_att = self.reparameterize(mu_p_att, logvar_p_att)
        z_from_att = z_t_from_att+z_p_from_att
        perm = torch.randperm(z_p_from_att.size(0))
        shuffled_z_p_from_att = z_p_from_att[perm]
        shuffled_z_from_att = shuffled_z_p_from_att + z_t_from_att

        ##############################################
        # Reconstruct inputs
        ##############################################


        img_from_img = self.decoder['resnet_features'](z_from_img)
        # img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        # att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)

        ##############################################
        # KL-Divergence
        ##############################################
        # KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
        #       + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))
        KLD_img = (0.5 * torch.sum(1 + logvar_t_img - mu_t_img.pow(2) - logvar_t_img.exp())) \
              + (0.5 * torch.sum(1 + logvar_p_img - mu_p_img.pow(2) - logvar_p_img.exp()))
        KLD_att = (0.5 * torch.sum(1 + logvar_t_att - mu_t_att.pow(2) - logvar_t_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_p_att - mu_p_att.pow(2) - logvar_p_att.exp()))
        KLD = KLD_img + KLD_att
        ##############################################
        # Distribution Alignment
        ##############################################
        # distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
        #                       torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
        distance_t = torch.sqrt(torch.sum((mu_t_img - mu_t_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_t_img.exp()) - torch.sqrt(logvar_t_att.exp())) ** 2, dim=1))
        distance_p = torch.sqrt(torch.sum((mu_p_img - mu_p_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_p_img.exp()) - torch.sqrt(logvar_p_att.exp())) ** 2, dim=1))
        distance = distance_t + distance_p
        distance = distance.sum()

        '''
            TODO 两个模态各增加一个辅助分类器，以计算辅助分类损失SC
        '''
        out_v = self.aux_cla['resnet_features'](shuffled_z_from_img)
        out_s = self.aux_cla['attributes'](shuffled_z_from_att)
        '''
            TODO 这里需要取得此batch的label
        '''
        loss_v = self.shuffle_classification_criterion(out_v, label)
        loss_s = self.shuffle_classification_criterion(out_s, label)
        shuffle_classification_loss = loss_v + loss_s
        
        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################

        f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])])

        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

        f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        loss = reconstruction_loss - beta * KLD

        print('-'*30)
        print('loss_v:{}, loss_s:{}, shuffle loss:{}'.format(loss_v, loss_s, shuffle_classification_loss))
        print('reconstruction_loss:', reconstruction_loss)
        print('KLD_img:{}, KLD_att:{}, KLD:{}'.format(KLD_img, KLD_att, KLD))
        print('reconstruction-beta*KLD:', loss)
        if cross_reconstruction_loss>0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor >0:
            loss += distance_factor*distance

        shuffle_factor = self.shuffle_factor
        loss += shuffle_factor*shuffle_classification_loss
        print('cross_reconstruction_loss:', cross_reconstruction_loss)
        print('distance_loss:', distance)
        print('total loss:', loss)
        print('-'*30)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_vae(self):

        losses = []
        '''
            FIXME 貌似这里定义的self.dataloader用不到，而且会报错：TypeError: object of type 'DATA_LOADER' has no len()
            因为self.dataset里定义了取batch的方法
        '''
        # self.dataloader = data.DataLoader(self.dataset,batch_size= self.batch_size,shuffle= True,drop_last=True)#,num_workers = 4)
        
        self.dataset.novelclasses =self.dataset.novelclasses.long().cuda()
        self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()
        #leave both statements
        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
        for epoch in range(0, self.nepoch ):
            self.current_epoch = epoch

            i=-1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i+=1
                # batch_label, [ batch_feature, batch_att]
                label, data_from_modalities = self.dataset.next_batch(self.batch_size)
                '''
                    TODO 这里为何要将requires_grad置为False？ 
                '''
                label= label.long().to(self.device)
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False

                loss = self.trainstep(data_from_modalities[0], data_from_modalities[1], label )

                if i%50==0:

                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+
                    ' | loss ' +  str(loss)[:5]   )

                if i%50==0 and i>0:
                    losses.append(loss)

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses

    def train_classifier(self, show_plots=False):

        if self.num_shots > 0 :
            print('================  transfer features from test to train ==================')
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')

        history = []  # stores accuracies


        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses


        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels'] # 离散的值
        novelclass_aux_data = self.dataset.novelclass_aux_data  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
        seenclass_aux_data = self.dataset.seenclass_aux_data

        '''
            TODO 下面这俩和cls_seenclasses除了在不同device上之外，有何区别？
            cls_seenclasses是用于作为传入map_label的参数的，
        '''
        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)


        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data['test_unseen'][
            'resnet_features']  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.data['test_seen'][
            'resnet_features']  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset.data['test_unseen']['labels']  # self.dataset.test_novel_label.to(self.device)

        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']


        # in ZSL mode:
        if self.generalized == False:
            # there are only 50 classes in ZSL (for CUB)
            # novel_corresponding_labels =list of all novel classes (as tensor)
            # test_novel_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:

            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)

            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)

            # for FSL, we train_seen contains the unseen class examples
            # for ZSL, train seen label is not used
            # if self.num_shots>0:
            #    train_seen_label = self.map_label(train_seen_label,cls_novelclasses)

            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)

            # map cls novelclasses last
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)


        if self.generalized:
            print('mode: gzsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            print('mode: zsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)


        clf.apply(models.weights_init)

        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################
            '''
                TODO 这里重参数为何要为False？
            '''
            self.reparameterize_with_noise = False

            # mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            mu1, var1, _, _ = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            test_novel_Y = test_novel_label.to(self.device)

            # mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            mu2, var2, _, _ = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            test_seen_Y = test_seen_label.to(self.device)

            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()

                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        '''
                            如果每个类的样本小于我们想采样的个数，则重复取，直到取到我们想采样的个数。
                            比如此类只有50个，我们要取200个，则重复四次
                        '''
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.cuda.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                         dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])


            # some of the following might be empty tensors if the specified number of
            # samples is zero :

            img_seen_feat,   img_seen_label   = sample_train_data_on_sample_per_class_basis(
                train_seen_feat,train_seen_label,self.img_seen_samples )

            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.img_unseen_samples )

            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                    novelclass_aux_data,
                    novel_corresponding_labels,self.att_unseen_samples )

            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seenclass_aux_data,
                seen_corresponding_labels, self.att_seen_samples)
            '''
                TODO 只使用解耦出的distilling特征来训练分类器
            '''
            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    # mu_, logvar_ = encoder(features)
                    mu_, logvar_, _, _ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.cuda.FloatTensor([])

            z_seen_img   = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])

            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])

            train_Z = [z_seen_img, z_unseen_img ,z_seen_att    ,z_unseen_att]
            train_L = [img_seen_label    , img_unseen_label,att_seen_label,att_unseen_label]

            # empty tensors are sorted out
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)

        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################

        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
                                    test_novel_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size, # 32
                                    self.generalized)
        # cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
        #                             test_novel_Y,
        #                             cls_seenclasses, cls_novelclasses,
        #                             self.num_classes, self.device, self.lr_cls, 0.5, 1,
        #                             self.classifier_batch_size, # 32
        #                             False)

        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()

            if self.generalized:

                print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
                k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))

                history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(),
                                torch.tensor(cls.H).item()])

            else:
                print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                history.append([0, torch.tensor(cls.acc).item(), 0])

        if self.generalized:
            return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(
                cls.H).item(), history
        else:
            return 0, torch.tensor(cls.acc).item(), 0, history
