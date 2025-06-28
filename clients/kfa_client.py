#!/usr/bin/env python
# coding: utf-8
import torch
import gc
from tqdm import tqdm
import sys
import os

# Dynamically add the FedKFA directory to the Python path
# This allows us to import its modules as if it were a top-level library
# This respects the fact that FedKFA was a standalone project.
try:
    # Get the absolute path of the current file's directory (clients/)
    clients_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (one level up)
    project_root = os.path.dirname(clients_dir)
    # Construct the path to the FedKFA directory
    fedkfa_path = os.path.join(project_root, 'FedKFA')
    # Add the FedKFA path to the system path if it's not already there
    if fedkfa_path not in sys.path:
        sys.path.append(fedkfa_path)
except NameError:
    # Fallback for some environments where __file__ is not defined
    fedkfa_path = os.path.join(os.getcwd(), 'FedKFA')
    if fedkfa_path not in sys.path:
        sys.path.append(fedkfa_path)


from clients.build import CLIENT_REGISTRY
from clients.base_client import Client
from utils.logging_utils import AverageMeter

# Now we can import directly from FedKFA's modules, because FedKFA is on the path
try:
    from kfac import KFACOptimizer
except ImportError as e:
    print(f"Could not import KFACOptimizer. Make sure the FedKFA directory exists and is accessible. Error: {e}")
    KFACOptimizer = None

import logging
logger = logging.getLogger(__name__)

@CLIENT_REGISTRY.register()
class KFAClient(Client):
    """
    Client for the FedKFA algorithm.
    It performs standard local training and then computes the Kronecker factors.
    """

    def local_train(self, global_epoch, **kwargs):
        """
        Performs the two-stage local update for FedKFA.
        Stage 1: Standard local training using SGD.
        Stage 2: Computation of Kronecker factors using K-FAC.
        """
        assert self.model is not None, "Model is not initialized!"
        if KFACOptimizer is None:
            raise ImportError("KFACOptimizer is not available. Please check your environment.")

        self.model.to(self.device)
        loss_meter = AverageMeter('Loss', ':.2f')

        # --- Stage 1: Local client training (standard SGD) ---
        for local_epoch in (epoch_pbar := tqdm(range(self.args.trainer.local_epochs), desc=f"Local Epochs (C_ID: {self.client_index})", leave=False)):
            for i, (images, labels) in enumerate(self.loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                
                # Using the base _algorithm for standard classification loss
                losses = self._algorithm(images, labels)
                loss = losses["cls"]

                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.item(), images.size(0))
            
            self.scheduler.step()
            epoch_pbar.set_postfix_str(f"Loss: {loss_meter.avg:.3f}")

        
        # --- Stage 2: Compute Kronecker Factors (K-FAC) ---
        # logger.info(f"[Client {self.client_index}] Computing Kronecker Factors")
        kfac_optim = KFACOptimizer(self.model)

        # Initialize dictionaries to accumulate factors, using module names as keys
        module_names = [kfac_optim.module2name[m] for m in kfac_optim.modules]
        accumulated_factors = {
            'm_aa': {name: torch.zeros_like(kfac_optim.model.get_submodule(name).weight, device=self.device) 
                     for name in module_names if hasattr(kfac_optim.model.get_submodule(name), 'weight')},
            'm_gg': {name: torch.zeros_like(kfac_optim.model.get_submodule(name).weight, device=self.device)
                     for name in module_names if hasattr(kfac_optim.model.get_submodule(name), 'weight')}
        }
        # A bit of a hack to initialize with correct sizes. Let's re-initialize properly.
        # Initializing accumulation dictionaries
        accumulated_factors = {'m_aa': {}, 'm_gg': {}}
        total_data_len = len(self.loader.dataset)


        # Iterate over data to compute and accumulate factors
        for i, (images, labels) in enumerate(self.loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.model.zero_grad()
            kfac_optim.zero_grad()

            # Forward pass
            results = self.model(images)
            
            # Compute loss
            loss = self.criterion(results["logit"], labels)

            # Backward pass computes gradients which KFAC hooks use to compute factors
            loss.backward()
            
            # Accumulate the factors, weighted by batch size
            batch_size = images.size(0)
            for module in kfac_optim.modules:
                name = kfac_optim.module2name[module]
                # Initialize tensors on first batch
                if i == 0:
                    accumulated_factors['m_aa'][name] = torch.zeros_like(kfac_optim.m_aa[module])
                    accumulated_factors['m_gg'][name] = torch.zeros_like(kfac_optim.m_gg[module])
                
                accumulated_factors['m_aa'][name] += (batch_size / total_data_len) * kfac_optim.m_aa[module]
                accumulated_factors['m_gg'][name] += (batch_size / total_data_len) * kfac_optim.m_gg[module]

        # The accumulated factors are the final ones
        kronecker_factors = {
            'm_aa': {name: v.to('cpu') for name, v in accumulated_factors['m_aa'].items()},
            'm_gg': {name: v.to('cpu') for name, v in accumulated_factors['m_gg'].items()},
        }

        # CRITICAL: Remove all hooks to prevent memory leaks
        for hook in kfac_optim.hooks:
            hook.remove()
        del kfac_optim # Explicitly delete the optimizer to help garbage collection

        # Clean up
        self.model.to('cpu')
        
        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
        }
        gc.collect()
        
        # IMPORTANT: We now return a third value, the kronecker_factors
        return self.model.state_dict(), loss_dict, kronecker_factors 