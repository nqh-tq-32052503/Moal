import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from utils.inc_net import SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet_adapt_AC
from utils.AC_net import SimpleCosineIncrementalNet, SimpleVitNet_AL, BiLoRAIncNet
from utils.data_manager import DataManager
from models.base import BaseLearner
from models.lora_drs_loss import AugmentedTripletLoss
from backbone.linears import CosineLinear
from utils.toolkit import target2onehot, tensor2numpy
import copy
from scipy.linalg import solve
from torch.utils.data import DataLoader, Dataset,TensorDataset,random_split

num_workers = 8

# 创建一个新的 全连接层
class SimpleNN(nn.Module):  
    def __init__(self, input_size, output_size,dtype=torch.float32):  
        super(SimpleNN, self).__init__()  
        # 初始化全连接层  
        # input_size: 输入特征的数量  
        # output_size: 输出特征的数量（即该层的神经元数量）  
        self.fc = nn.Linear(input_size, output_size)  
  
    def forward(self, x):  
        if x.dtype != self.fc.weight.dtype:  
            x = x.to(dtype=self.fc.weight.dtype)  
        x = self.fc(x)  
        return x  
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["backbone_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
            # self._network = SimpleVitNet(args, True)
        self._network = BiLoRAIncNet(args, True)
        self.model_type = self._network.model_type
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.progressive_lr = args["progressive_lr"]
        self.model_hidden = args["Hidden"] 
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.R = None
        self._means = []
    
    def init_fc_ac(self):
        self._network.init_ac_fc()

    def init_from_previous_checkpoint(self, previous_task_checkpoint="./checkpoint.pth"):
        previous_checkpoint = torch.load(previous_task_checkpoint, map_location='cpu', weights_only=False)
        self._network = BiLoRAIncNet(self.args, True)
        self._network.load_state_dict(previous_checkpoint['network'], strict=False)
        self._network.current_task = previous_checkpoint["current_network_task"]
        self._network.load_fc(previous_checkpoint['network'])
        self._network.load_ac(previous_checkpoint['network'])
        self._network.to(self._device)
        self._old_network = self._network.copy().freeze().to(self._device)
        self.old_network_module_ptr = self._old_network
        self._known_classes = previous_checkpoint['known_classes']
        self._total_classes = previous_checkpoint['total_classes']
        self._cur_task = previous_checkpoint['cur_task']
        self._means = [torch.tensor(mean).numpy() for mean in previous_checkpoint['means']]
        self.R = previous_checkpoint['R'].to(self._device)
        print("Loaded previous task checkpoint from {}".format(previous_task_checkpoint))

    def save_after_task(self, path="./checkpoint.pth"):
        state = {
            'network': self._network.state_dict(),
            "current_network_task": self._network.current_task,
            'known_classes': self._known_classes,
            'total_classes': self._total_classes,
            'cur_task': self._cur_task,
            'means': [torch.from_numpy(np.array(mean)) for mean in self._means],
            'R': self.R,
        }
        torch.save(state, path)
        print("Model saved to {}".format(path))
    def next_task(self):
        try:
            self._network.update_task()
            print("Network updated for new task.")
        except:
            pass

    def after_task(self):
        self._known_classes = self._total_classes
        # calculate before update the old_model

        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def incremental_train(self, data_manager: DataManager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        print("Train dataset size: {}".format(len(train_dataset)))
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        print("Finish one task ")
        

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        self._network.to(self._device)

        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                      weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self._network.list_ac = nn.ModuleList()
            self._network.update_fc(cosine_fc=True)
            self._network.update_fc()
            self._network.update_task()

        else:
            self._network.update_fc(cosine_fc=True)
            self._network.update_fc()
            self._network.update_task()
            for i in range(len(self._network.list_ac)):
                for param in self._network.list_ac[i].parameters():
                    param.requires_grad = False
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            # if total_params != total_trainable_params:
            #     for name, param in self._network.named_parameters():
            #         if param.requires_grad:
            #             print(name, param.numel())
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.progressive_lr,
                                      weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.progressive_lr,
                                        weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['progreesive_epoch'],
                                                             eta_min=self.min_lr)
            self._progreessive_train(train_loader, test_loader, optimizer, scheduler)

        if self._cur_task == 0:
            self._compute_means()
            self._network.to(self._device)
            # AL training process
            self.cls_align(train_loader, self._network)
        else:
            self._compute_means()
            self.cali_prototye_model(train_loader)
            self._compute_relations()
            self._build_feature_set()
            self._network.to(self._device)
            # AL training process
            self.IL_align(train_loader, self._network)
            self.cali_weight(self._feature_trainset, self._network)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        print("Initial training for the first task.")
        ranking_criterion = AugmentedTripletLoss(margin=1.0).to(self._device)
        ranking_lambda = 0.05
        for epoch in range(self.args['tuned_epoch']):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for train_batch in tqdm(train_loader):
                _, inputs, targets = train_batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                if self.model_type == 'bilora':
                    outputs = self._network(inputs, task=self._cur_task)
                    logits = outputs["logits"]
                else:
                    outputs = self._network(inputs)
                    logits = outputs["logits"]

                loss = F.cross_entropy(logits, targets)
                ATL_loss = ranking_criterion(outputs['features'], targets, [])
                loss = loss + ranking_lambda * ATL_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader, task=self._cur_task)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            print(info)

    def _progreessive_train(self, train_loader, test_loader, optimizer, scheduler):
        print("Progressive training for task {}".format(self._cur_task))
        self.output_caches = []
        self.label_caches = []
        ranking_criterion = AugmentedTripletLoss(margin=1.0).to(self._device)
        ranking_lambda = 0.05
        EMA_model = self._network.copy().freeze()
        alpha = self.args['alpha']

        for epoch in range(self.args['progreesive_epoch']):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            run_cache = epoch == self.args['progreesive_epoch'] - 1
            for train_batch in tqdm(train_loader):
                _, inputs, targets = train_batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                if self.model_type == 'bilora':
                    outputs = self._network(inputs, task=self._cur_task)
                    # logits = outputs["train_logits"]
                    logits = outputs["logits"]
                else:
                    outputs = self._network(inputs)
                    # logits = outputs["train_logits"]
                    logits = outputs["logits"]
                if run_cache:
                    self.output_caches.append(outputs)
                    self.label_caches.append({"input" : inputs, "label" : targets})
                loss_ce = F.cross_entropy(logits, targets)
                ATL_loss = ranking_criterion(outputs['features'], targets, self._means)
                loss = loss_ce + ranking_lambda * ATL_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            for param, ema_param in zip(self._network.backbone.parameters(), EMA_model.backbone.parameters()):
                ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_ac_train_accuracy(self._network, test_loader, task=self._cur_task)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['progreesive_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            print(info)

        for param, ema_param in zip(EMA_model.backbone.parameters(),
                                    self._network.backbone.parameters()):
            ema_param.data =  param.data
        print("Cache size: {}".format(len(self.output_caches)))
        print(info)

    def cls_align(self, trainloader, model: BiLoRAIncNet):
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model

        embedding_list = []
        label_list = []

        # AL training process
        model = model.eval()
        auto_cor_size = sum([ac_model.fc[-1].weight.size(1) for ac_model in model.list_ac])
        auto_cor = torch.zeros(auto_cor_size, auto_cor_size).to(self._device)
        crs_cor = torch.zeros(auto_cor_size, self._total_classes).to(self._device)
        print("Starting class alignment...")
        with torch.no_grad():
            pbar = tqdm(enumerate(trainloader), desc='Alignment', total=len(trainloader), unit='batch')
            for i, batch in pbar:
                (_, data, label) = batch
                images = data.to(self._device)
                target = label.to(self._device)

                label_list.append(target.cpu())
                if self.model_type == 'bilora':
                    feature = model(images, task=self._cur_task)["features"]
                else:
                    feature = model(images)["features"]
                
                new_activation = model.list_ac[-1].fc[:2](feature)

                embedding_list.append(new_activation.cpu())

                label_onehot = F.one_hot(target, self._total_classes).float()
                auto_cor += torch.t(new_activation) @ new_activation
                crs_cor += torch.t(new_activation) @ (label_onehot)

        embedding_list = torch.cat(embedding_list, dim=0)
        print("Embedding shape: ", embedding_list.shape)
        label_list = torch.cat(label_list, dim=0)
        print("Label shape", label_list.shape)
        Y = target2onehot(label_list, self._total_classes)
        print("One-hot label shape: ", Y.shape)
        ridge = self.optimise_ridge_parameter(embedding_list, Y)
        print("gamma {}".format(ridge))

        print('numpy inverse')
        R = np.asmatrix(auto_cor.cpu().numpy() + ridge * np.eye(model.list_ac[-1].fc[-1].weight.size(1))).I
        R = torch.tensor(R).float().to(self._device)
        Delta = R @ crs_cor
        model.list_ac[-1].fc[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9 * Delta.float()))
        self.R = R
        del R

    def optimise_ridge_parameter(self, Features, Y):
        print('Optimising ridge parameter...')
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print('selected lambda =',ridge)
        return ridge

    # def update_separate_fc_optimized(self, W, R, A, L):
    #     inner_diff = L - torch.matmul(A, W)
    #     update_term = torch.matmul(torch.matmul(R, A.t()), inner_diff)
    #     return W + update_term
    def update_separate_fc(self, W, R, A, L):
        embed_dim, num_classes = W.shape
        num_class_per_task = self.args['increment']
        num_tasks = num_classes // num_class_per_task
        outputs = []
        for t in range(num_tasks):
            start_class = t * num_class_per_task
            end_class = start_class + num_class_per_task
            W_t = W[:, start_class:end_class]
            L_t = L[:, start_class:end_class]
            Q_t = W_t + R @ A.T @ (L_t - A @ W_t)
            outputs.append(Q_t)
        return torch.cat(outputs, dim=1)

    def IL_align(self, trainloader, model: BiLoRAIncNet):
        print("Incremental class alignment (Knowledge Memorization)...")
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model

        # AL training process
        model = model.eval()
        W = torch.cat([ac_model.fc[-1].weight.t().float() for ac_model in model.list_ac], dim=1)
        # W = (model.list_ac[-1].fc[-1].weight.t()).float()
        R = copy.deepcopy(self.R.float())

        with torch.no_grad():
            # pbar = tqdm(enumerate(trainloader), desc='Alignment', total=len(trainloader), unit='batch')
            for i in range(len(self.output_caches)):
                # (_, data, label) = batch
                
                # images = data.to(self._device)
                target = self.label_caches[i]["label"]
                # if self.model_type == 'bilora':
                #     feature = model(images, task=self._cur_task)["features"]
                # else:
                #     feature = model(images)["features"]
                feature = self.output_caches[i]["features"]
                new_activation = model.list_ac[-1].fc[:2](feature)
                label_onehot = F.one_hot(target, self._total_classes).float()

                R = R - R @ new_activation.t() @ torch.pinverse(
                    torch.eye(new_activation.size(0)).to(self._device) +
                    new_activation @ R @ new_activation.t()) @ new_activation @ R

                # W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)
                W = self.update_separate_fc(W=W, R=R, A=new_activation, L=label_onehot)
            print("Knowledge Memorization completed.")
            print("Updated weight matrix W shape: {}".format(W.shape))
            print("Updated correlation matrix R shape: {}".format(R.shape))

        print('numpy inverse')
        # model.list_ac[-1].fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        N = len(model.list_ac)
        W_t = torch.t(W.float()) # Nếu W đã là (10N, 5000) thì không cần .t() nữa, 
                        # vì nó đã khớp định dạng [out, in] của PyTorch

        # Chia W thành N phần, mỗi phần có 10 hàng
        w_parts = torch.split(W_t, split_size_or_sections=10, dim=0)
        for i in range(N):
            model.list_ac[i].fc[-1].weight = torch.nn.Parameter(w_parts[i])
        self.R = R
        del R


    def _compute_means(self):
        print("Computing class means and covariance matrices...")
        self.vectors_train = []
        self.labels_train = []
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self.extract_prototype(idx_loader)
                self.vectors_train.append(vectors)
                self.labels_train.append([class_idx] * len(vectors))
                class_mean = np.mean(vectors, axis=0)
                self._means.append(class_mean)
            print("Generating pseudo-features for old classes from relations...")
    def _compute_relations(self):
        print("Computing class relations...")
        old_means = np.array(self._means[:self._known_classes])
        print("Old means shape: {}".format(old_means.shape))
        new_means = np.array(self._means[self._known_classes:])
        print("New means shape: {}".format(new_means.shape))
        self._relations = np.argmax((old_means / np.linalg.norm(old_means, axis=1)[:, None]) @ (
                new_means / np.linalg.norm(new_means, axis=1)[:, None]).T, axis=1) + self._known_classes
        print("Class relations: {}".format(self._relations))

    def extract_prototype(self, loader):
        self._network.eval()
        vectors, targets = [], []
        print("Extracting prototypes...")
        for batch in tqdm(loader):
            _, _inputs, _targets = batch
            _targets = _targets.numpy()
            if self.model_type == 'bilora':
                tensor_features = self._network(_inputs.to(self._device), task=self._cur_task)["features"]
            else:
                tensor_features = self._network(_inputs.to(self._device))["features"]
            _vectors = tensor2numpy(tensor_features)
            vectors.append(_vectors)
            targets.append(_targets)
        return_vectors = np.concatenate(vectors)
        print('Extracted vectors shape:', return_vectors.shape, return_vectors.dtype)
        return_targets = np.concatenate(targets)
        print('Extracted targets shape:', return_targets.shape, return_targets.dtype)
        return return_vectors, return_targets

    def _build_feature_set(self):
        print("Building feature dataset...")
        # self.vectors_train = []
        # self.labels_train = []
        print("Extract prototypes for known classes...")
        # for class_idx in range(self._known_classes, self._total_classes):
        #     data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
        #                                                                source='train',
        #                                                                mode='test', ret_data=True)
        #     idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
        #     vectors, _ = self.extract_prototype(idx_loader)
        #     self.vectors_train.append(vectors)
        #     self.labels_train.append([class_idx] * len(vectors))
        # print("Generating pseudo-features for old classes from relations...")
        for class_idx in range(0, self._known_classes):
            new_idx = self._relations[class_idx]
            self.vectors_train.append(
                self.vectors_train[new_idx - self._known_classes] - self._means[new_idx] + self._means[class_idx])
            self.labels_train.append([class_idx] * len(self.vectors_train[-1]))

        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        print("Total feature dataset size: {}".format(self.vectors_train.shape[0]))
        print("Feature dataset dimension: {}".format(self.vectors_train.shape[1]))
        print("Label dataset size: {}".format(self.labels_train.shape[0]))
        print("Label dataset classes: {}".format(np.unique(self.labels_train)))
        self._feature_trainset = FeatureDataset(self.vectors_train, self.labels_train)
        self._feature_trainset = DataLoader(self._feature_trainset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)


    def cali_weight(self, cali_pseudo_feature, model: BiLoRAIncNet):
        print("Calibrating classifier weights (Knowledge Rumination - Selective reinforcement of old task knowledge)...")
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model

        # AL training process
        model = model.eval()

        W = torch.cat([ac_model.fc[-1].weight.t().float() for ac_model in model.list_ac], dim=1)
        R = copy.deepcopy(self.R.float())

        with torch.no_grad():
            pbar = tqdm(enumerate(cali_pseudo_feature), desc='Alignment', total=len(cali_pseudo_feature), unit='batch')
            for i, batch in pbar:
                (_, data, label) = batch
                features = data.to(self._device)
                target = label.to(self._device)

                new_activation = model.list_ac[-1].fc[:2](features.float())
                label_onehot = F.one_hot(target, self._total_classes).float()

                # 获取wrong prediction的索引
                output = model.list_ac[-1].fc[-1](new_activation)
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                false_indices = (correct == False).view(-1).nonzero(as_tuple=False) # Indicator matrix M

                new_activation = new_activation[false_indices[:, 0]]
                label_onehot = label_onehot[false_indices[:, 0]]

                R = R - R @ new_activation.t() @ torch.pinverse(
                    torch.eye(new_activation.size(0)).to(self._device) +
                    new_activation @ R @ new_activation.t()) @ new_activation @ R

                # W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)
                W = self.update_separate_fc(W=W, R=R, A=new_activation, L=label_onehot)

        print('numpy inverse')
        # model.list_ac[-1].fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        N = len(model.list_ac)
        W_t = torch.t(W.float()) # Nếu W đã là (10N, 5000) thì không cần .t() nữa, 
                        # vì nó đã khớp định dạng [out, in] của PyTorch

        # Chia W thành N phần, mỗi phần có 10 hàng
        w_parts = torch.split(W_t, split_size_or_sections=10, dim=0)

        for i in range(N):
            model.list_ac[i].fc[-1].weight = torch.nn.Parameter(w_parts[i])
        self.R = R
        del R

    def cali_prototye_model(self,train_loader):
        print("Calibrating prototype model (Prototype correction - Knowledge Rumination)...")
        with torch.no_grad():
            old_vectors, vectors, targets = [], [], []

            for i in range(len(self.output_caches)):
                images = self.label_caches[i]["input"]

                
                if self.model_type == 'bilora':
                    old_tensor_features = self.old_network_module_ptr(images, task=self._cur_task)["features"]
                    # new_tensor_features = self._network(images, task=self._cur_task)["features"]
                else:
                    old_tensor_features = self.old_network_module_ptr(images)["features"]
                    # new_tensor_features = self._network(images)["features"]
                new_tensor_features = self.output_caches[i]["features"]
                old_feature = tensor2numpy(old_tensor_features)
                feature = tensor2numpy(new_tensor_features)

                old_vectors.append(old_feature)
                vectors.append(feature)  
        E_old = np.concatenate(old_vectors)
        E_new = np.concatenate(vectors)
        # A = E_old.copy()
        # B = E_new.copy()
        # ATA = A.T @ A
        # ATB = A.T @ B
        # W = solve(ATA, ATB)  # 使用 scipy.linalg.solve 求解线性方程组
        # 准备训练数据 
        X_tensor = torch.from_numpy(E_old).to(torch.float32)

        y_tensor = torch.from_numpy(E_new).to(torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 划分训练集和测试集
        total_size = len(dataset)
        train_size = int(0.9 * total_size)  # 90% 为训练集
        test_size = total_size - train_size  # 剩余的 10% 为测试集
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # 构造模型参数 && 模型初始化 
        in_features = E_old[0].shape[0]  # 输入维度
        out_dim = E_new[0].shape[0]       # 输出维度 
        calimodel = SimpleNN(in_features,out_dim)
        calimodel = calimodel.to(self._device)
        #calimodel.to(torch.float32) 
        # 设置 学习率 优化器 
        optimizer = optim.SGD(calimodel.parameters(), momentum=0.9, lr=0.01,
            weight_decay=0.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,
            eta_min=0)     
        prog_bar = tqdm(range(50))
        
        # 保存训练过程中的最好模型
        best_loss = float('inf')  # 初始化为无穷大，假设损失越小越好  
        best_model_wts = None  
        print("开始 修正 prototype")
        for _, epoch in enumerate(prog_bar):
            calimodel.train()
            running_loss = 0.0 
            for i, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                #logits = calimodel(inputs)["logits"]
                logits = calimodel(inputs)
                criterion = nn.MSELoss()
                # 计算二次范数损失
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            # 计算每个epoch的平均损失 
            scheduler.step() 
            calimodel.eval()
            test_loss = 0.0
            #correct = 0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = calimodel(inputs)
                    criterion = nn.MSELoss()
                    test_loss += criterion(logits, targets).item() * inputs.size(0)
    
            test_loss /= len(test_dataset)
            if test_loss < best_loss:  
                #print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Best model updated!')  
                best_loss = test_loss  
                best_model_wts = copy.deepcopy(calimodel.state_dict()) 
        print("best_loss: {}".format(best_loss))
        # 选取最好参数 
        calimodel.load_state_dict(best_model_wts)  
        calimodel.eval()
        X_test = torch.from_numpy(np.array(self._means)[:self._known_classes]).to(torch.float32)
        # Y_test = X_test @ W  # 预测输出
        # Y_test = calimodel(X_test.to(self._device))["logits"]
        Y_test = calimodel(X_test.to(self._device))
        Y_test = Y_test.to("cpu")  
        Y_test = Y_test.detach().numpy().tolist()
        self._means[:self._known_classes] = Y_test



class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label





