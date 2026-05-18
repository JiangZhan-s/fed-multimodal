
import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch


from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class ClientFedAvg(object):
    def __init__(
        self, 
        args, 
        device, 
        criterion, 
        dataloader, 
        model, 
        label_dict=None,
        num_class=None
    ):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        
    def get_parameters(self):
        # Return model parameters
        return self.model.state_dict()
    
    def get_model_result(self):
        # Return model results
        return self.result
    
    def get_test_true(self):
        # Return test labels
        return self.test_true
    
    def get_test_pred(self):
        # Return test predictions
        return self.test_pred
    
    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth

    def update_weights(self):
        # Set mode to train model
        self.model.train()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        self.gate_stats = None
        gate_stat_sums = {
            "mean_gate_acc": 0.0,
            "mean_gate_gyro": 0.0,
            "gate_entropy": 0.0,
        }
        gate_stat_samples = 0
        
        # optimizer
        if self.args.fed_alg in ['fed_avg', 'fed_opt', 'fed_rant_lite']:
            optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5
            )
        else:
            optimizer = FedProxOptimizer(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
                mu=self.args.mu
            )
            
        # last global model
        last_global_model = copy.deepcopy(self.model)
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20: continue
                self.model.zero_grad()
                optimizer.zero_grad()
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                    
                    # forward
                    outputs, _ = self.model(
                        x_a.float(), x_b.float(), l_a, l_b
                    )
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    # forward
                    outputs, _ = self.model(
                        x.float(), l
                    )
                
                if not self.multilabel: 
                    outputs = torch.log_softmax(outputs, dim=1)
                    
                # backward
                loss = self.criterion(outputs, y)
                total_loss = loss
                if (
                    self.args.modality == "multimodal"
                    and getattr(self.args, "att", False)
                    and getattr(self.args, "att_name", None) == "reliability_gate"
                ):
                    gate_stats = getattr(self.model, "last_gate_stats", None)
                    if gate_stats is not None:
                        batch_samples = int(y.shape[0])
                        for key in gate_stat_sums:
                            gate_stat_sums[key] += float(gate_stats[key]) * batch_samples
                        gate_stat_samples += batch_samples

                    gate_entropy = getattr(self.model, "last_gate_entropy", None)
                    if gate_entropy is not None and getattr(self.args, "gate_entropy_reg", 0.0) > 0:
                        total_loss = loss - self.args.gate_entropy_reg * gate_entropy

                # backward
                total_loss.backward()
                
                # clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    10.0
                )
                optimizer.step()
                
                # save results
                if not self.multilabel: 
                    self.eval.append_classification_results(
                        y, 
                        outputs, 
                        loss
                    )
                else:
                    self.eval.append_multilabel_results(
                        y, 
                        outputs, 
                        loss
                    )
                
        # epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()

        if gate_stat_samples > 0:
            self.gate_stats = {
                key: value / gate_stat_samples
                for key, value in gate_stat_sums.items()
            }

    def get_gate_stats(self):
        return self.gate_stats
