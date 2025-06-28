#!/usr/bin/env python
# coding: utf-8
import torch
import copy
from collections import defaultdict
import logging

from servers.build import SERVER_REGISTRY
from servers.base import Server

logger = logging.getLogger(__name__)

@SERVER_REGISTRY.register()
class KFAServer(Server):
    """
    Server for the FedKFA algorithm.
    It aggregates client models using Kronecker-factored approximations of the
    Fisher Information Matrix, based on Riemannian geometry principles.
    This implementation uses Newton's method to solve the aggregation equation,
    matching the logic from the original research script.
    """

    def __init__(self, args):
        super().__init__(args)
        # Damping parameter for matrix inversion for numerical stability
        self.damping = args.server.kfa_damping
        # Steps for the iterative Newton's method solver. 10000 is from the original script.
        self.newton_steps = getattr(args.server, 'kfa_newton_steps', 10000)

    def aggregate(self, local_state_dicts, kronecker_factors_list, client_weights, **kwargs):
        """
        Aggregates models and Kronecker factors from clients.
        """
        # Get the keys for parameters that are handled by K-FAC
        # We assume all clients compute factors for the same set of modules
        kfac_modules = kronecker_factors_list[0]['m_aa'].keys()
        
        # Aggregate model weights using Newton's method
        new_global_state_dict = self._aggregate_weights(local_state_dicts, kronecker_factors_list, client_weights, kfac_modules)
        
        return new_global_state_dict

    def _aggregate_weights(self, local_state_dicts, kronecker_factors_list, client_weights, kfac_modules):
        """Aggregate model weights using the K-FAC Newton's method procedure."""
        
        # Initialize a new state_dict with a deepcopy of the first client's state_dict structure
        pristine_state_dict = local_state_dicts[0]
        new_global_state_dict = copy.deepcopy(pristine_state_dict)
        # Zero out all parameters, we will fill them
        for k in new_global_state_dict:
            new_global_state_dict[k].zero_()

        # Handle K-FAC modules
        global_transformed_weights = defaultdict(lambda: 0)

        for client_weight, state_dict, factors in zip(client_weights, local_state_dicts, kronecker_factors_list):
            for mod_key in kfac_modules:
                W = self._get_patched_weight(state_dict, mod_key)
                A_k = factors['m_aa'][mod_key]
                G_k = factors['m_gg'][mod_key]

                # Transform weights: G_k @ W @ A_k
                transformed_W = G_k @ W @ A_k
                global_transformed_weights[mod_key] += client_weight * transformed_W
        
        # Compute final aggregated weights for K-FAC modules using Newton's method
        for mod_key in kfac_modules:
            # This is Z in the equation: Z = E[G_k @ W_k @ A_k]
            agg_transformed_W = global_transformed_weights[mod_key]
            # Add assert to help linter understand the type is a Tensor, as defaultdict can return 0.
            assert isinstance(agg_transformed_W, torch.Tensor), f"Expected a Tensor but got {type(agg_transformed_W)}"

            # These are the lists of Ak and Gk for all clients
            Ak_list = [factors['m_aa'][mod_key] for factors in kronecker_factors_list]
            Gk_list = [factors['m_gg'][mod_key] for factors in kronecker_factors_list]
            
            # Solve for the new global weight matrix W_g using Newton's method
            # Solves E[G_k @ W_g @ A_k] = Z
            device = agg_transformed_W.device
            final_W = self._newton_solve(client_weights, Ak_list, Gk_list, agg_transformed_W, device)
            
            original_shape = pristine_state_dict[f'{mod_key}.weight'].shape
            self._set_patched_weight(new_global_state_dict, mod_key, final_W, original_shape)

        # Handle non-K-FAC parameters using standard FedAvg
        all_param_keys = new_global_state_dict.keys()
        kfac_param_keys = self._get_kfac_param_keys(kfac_modules, pristine_state_dict)
        non_kfac_param_keys = set(all_param_keys) - set(kfac_param_keys)

        for key in non_kfac_param_keys:
            for client_weight, state_dict in zip(client_weights, local_state_dicts):
                new_global_state_dict[key] += client_weight * state_dict[key]

        return new_global_state_dict

    def _newton_solve(self, client_weights, Ak_list, Gk_list, Z, device):
        """
        Solves the equation E[G_k @ W_g @ A_k] = Z for W_g using iterative gradient descent.
        This logic is identical to the `newton_solve_auto_gradient` function in the original script.
        """
        # Initial guess for W_g (named HatM in original script) using the expectation approximation method.
        sum_A = sum(w * A for w, A in zip(client_weights, Ak_list))
        sum_G = sum(w * G for w, G in zip(client_weights, Gk_list))
        # Add assert to help linter, as sum() on an empty sequence returns 0.
        assert isinstance(sum_A, torch.Tensor)
        assert isinstance(sum_G, torch.Tensor)
        
        # Add damping for stability during inversion for the initial guess
        I_a = torch.eye(sum_A.size(0), device=device)
        I_g = torch.eye(sum_G.size(0), device=device)
        inv_sum_A = torch.inverse(sum_A + self.damping * I_a)
        inv_sum_G = torch.inverse(sum_G + self.damping * I_g)
        
        HatM = inv_sum_G @ Z @ inv_sum_A

        objective = torch.nn.MSELoss()
        ulr = 1.0  # Update learning rate, from original implementation

        for i in range(self.newton_steps):
            HatM.requires_grad_(True)
            if HatM.grad is not None:
                HatM.grad.zero_()
            
            # Calculate E[G_k @ HatM @ A_k]
            MuZ = sum(w * G @ HatM @ A for w, G, A in zip(client_weights, Gk_list, Ak_list))
            
            loss = objective(MuZ, Z)
            loss.backward()
            
            with torch.no_grad():
                gradient = ulr * HatM.grad.data
                HatM = HatM - gradient

        return HatM.detach()

    def _get_patched_weight(self, state_dict, mod_key):
        """Patches weight and bias of a layer into a single 2D matrix."""
        weight = state_dict[f'{mod_key}.weight']
        
        # Reshape conv weights to 2D
        if len(weight.shape) > 2: # It's a Conv layer
            weight_2d = weight.view(weight.shape[0], -1)
        else: # It's a Linear layer
            weight_2d = weight

        if f'{mod_key}.bias' in state_dict:
            bias = state_dict[f'{mod_key}.bias']
            return torch.cat([weight_2d, bias.unsqueeze(1)], dim=1)
        else:
            # If no bias, just return the reshaped weight.
            return weight_2d
    
    def _set_patched_weight(self, state_dict, mod_key, patched_weight, original_shape):
        """Splits the patched weight back into weight and bias, reshaping if necessary."""
        if f'{mod_key}.bias' in state_dict:
            weight_2d = patched_weight[:, :-1]
            state_dict[f'{mod_key}.weight'] = weight_2d.view(original_shape)
            state_dict[f'{mod_key}.bias'] = patched_weight[:, -1]
        else:
            # If no bias, the patched_weight is the weight itself.
            state_dict[f'{mod_key}.weight'] = patched_weight.view(original_shape)
            
    def _get_kfac_param_keys(self, kfac_modules, pristine_state_dict):
        """Gets all state_dict keys corresponding to K-FAC modules, checking for bias existence."""
        keys = []
        for mod_key in kfac_modules:
            keys.append(f'{mod_key}.weight')
            if f'{mod_key}.bias' in pristine_state_dict:
                keys.append(f'{mod_key}.bias')
        return keys 