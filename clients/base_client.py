#!/usr/bin/env python
# coding: utf-8
import copy
import time
from collections import defaultdict
from typing import Callable, Dict, Tuple, Union, List, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import gc
import tqdm

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from utils.logging_utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY


@CLIENT_REGISTRY.register()
class Client():

    def __init__(self, args, client_index, model: Optional[nn.Module] = None, loader=None):
        self.args = args
        self.client_index = client_index
        # self.loader = loader  
        self.model = model
        self.global_model =  model
        self.criterion = nn.CrossEntropyLoss()
        return

    def setup(self, state_dict, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):

        self._update_model(state_dict)
        self.device = device

        if self.args.dataset.num_instances > 0:
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)   
        else:
            train_sampler = None
        self.loader =  DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
                                   num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        
        assert self.model is not None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum,
                                   weight_decay=self.args.optimizer.wd)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                     lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
    
        self.trainer = trainer
        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
        if global_epoch == 0:
            logger.info(f"Class counts : {self.class_counts}")
        

    def _update_model(self, state_dict):
        assert self.model is not None
        self.model.load_state_dict(state_dict)

    def _update_global_model(self, state_dict):
        assert self.global_model is not None
        self.global_model.load_state_dict(state_dict)

    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : " + str(len(self.loader.dataset)) if self.loader and self.loader.dataset else ""}')

    def get_weights(self, epoch=None):

        weights = {
            "cls": 1
        }
        
        return weights

    def local_train(self, global_epoch, **kwargs):
        assert self.model is not None
        self.global_epoch = global_epoch

        self.model.to(self.device)
        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        # logger.info(f"[Client {self.client_index}] Local training start")

        self.weights = self.get_weights(epoch=global_epoch)

        if global_epoch % 50 == 0:
            print(self.weights)

        for local_epoch in (epoch_pbar := tqdm.tqdm(range(self.args.trainer.local_epochs), desc=f"Local Epochs (C_ID: {self.client_index})", leave=False)):
            end = time.time()

            for i, (images, labels) in enumerate(self.loader):
                    
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    losses = self._algorithm(images, labels)
                    loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    scaler.step(self.optimizer)
                    scaler.update()

                except Exception as e:
                    print(e)

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()
            
            epoch_pbar.set_postfix_str(f"Loss: {loss_meter.avg:.3f}")
            self.scheduler.step()
        
        # logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, Loss: {loss_meter.avg:.3f}")
        
        self.model.to('cpu')

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
        }
        gc.collect()
        
        return self.model.state_dict(), loss_dict
    

    def _algorithm(self, images, labels, ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        assert self.model is not None
        results = self.model(images)
        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        del results
        return losses


