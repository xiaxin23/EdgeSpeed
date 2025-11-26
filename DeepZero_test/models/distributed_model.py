import torch
import sys
from functools import reduce
from tools import *

sys.path.append(".")
from algorithm.prune import extract_mask, custom_prune

class DistributedCGEModel(object):
    def __init__(self, device, network_init_func, loss_function, param_name_to_module_id=None, init_file=None, pruned=False, feature_reuse=True) -> None:
        super().__init__()
        self.device = device
        self.pruned = pruned
        self.feature_reuse = feature_reuse
        self.network = network_init_func().to(device)
        self.param_name_to_module_id = param_name_to_module_id
        self.current_pertub = {}
        if init_file is not None:
            init_state_dict = torch.load(init_file, map_location=self.device)
            mask = extract_mask(init_state_dict)
            custom_prune(self.network, mask)
            self.network.load_state_dict(init_state_dict)
        self.loss_function = loss_function
        self.network.requires_grad_(False)

    @torch.no_grad()
    def calculate_grads(self, x_rref, y_rref, cge_step_size, random_seed):
        assert hasattr(self, 'instruction')
        set_seed(random_seed)
        x, y = x_rref.to_here().to(self.device), y_rref.to_here().to(self.device)
        fxs = [x]
        with torch.no_grad():
            fxs += self.network(x, return_interval = self.feature_reuse) if self.feature_reuse else [self.network(x, return_interval = self.feature_reuse)]
            base = self.loss_function(fxs[-1], y)
        grads = torch.zeros(self.instruction.size(0), device=self.device)
        for i, name_id in enumerate(self.instruction):
            self.perturb_block_param(self.param_names[name_id], cge_step_size)
            starting_id = self.param_name_to_module_id(self.param_names[name_id][0]) if self.feature_reuse else 0
            with torch.no_grad():
                fx = self.network(fxs[starting_id], starting_id=starting_id)
            self.perturb_block_param(self.param_names[name_id], cge_step_size, reset=True)
            grads[i] = self.loss_function(fx, y)
        grads = (grads - base) / cge_step_size
        grads = grads.cpu()
        # print(grads)
        # print(self.instruction)
        # new = torch.cat([self.instruction.unsqueeze(1), grads.unsqueeze(1)], dim=1)
        # print(new)
        return torch.cat([self.instruction.unsqueeze(1), grads.unsqueeze(1)], dim=1)

    def synchronize(self, params_to_be_updated_rref):
        params_to_be_updated = params_to_be_updated_rref.to_here().to(self.device)
        current_cnt = 0
        for name_id, keys in enumerate(self.param_names):
            for key in keys:
                names, attr = key.split('.')[:-1], key.split('.')[-1]
                module = self.get_module_by_name(self.network, names)
                param = getattr(module, attr).flatten()
                param_to_be_updated = params_to_be_updated[current_cnt:current_cnt+int(param.numel())].reshape_as(param)
                param.data.copy_(param_to_be_updated)
                current_cnt += int(param.numel())

    def set_params_to_be_perturbed(self, instruction_rref, names):
        self.instruction = instruction_rref.to_here().long()
        self.set_param_names(names)

    def set_param_names(self, names):
        # if not self.pruned:
        #     names = [name.replace('_orig', '') for name in names]
        self.param_names = names

    def perturb_a_param(self, key, idx, cge_step_size, reset=False):
        names, attr = key.split('.')[:-1], key.split('.')[-1]
        module = self.get_module_by_name(self.network, names)
        param = getattr(module, attr).flatten()
        if reset:
            param[idx] -= cge_step_size
        else:
            param[idx] += cge_step_size
    
    def perturb_block_param(self, keys, cge_step_size, reset=False):
        for key in keys:
            names, attr = key.split('.')[:-1], key.split('.')[-1]
            module = self.get_module_by_name(self.network, names)
            param = getattr(module, attr).flatten()
            if reset:
                param -= self.current_pertub[key]
                del self.current_pertub[key]
            else:
                perturb = torch.randn_like(param)
                perturb /= (torch.norm(perturb) + 1e-8)
                perturb *= cge_step_size
                self.current_pertub[key] = perturb
                # print("key: {}, perturb: {}".format(key, perturb.reshape_as(getattr(module, attr))))
                param += perturb
            # print('-'*50)
            # print(another)
            # print(param)
            # names, attr = key.split('.')[:-1], key.split('.')[-1]
            # module = self.get_module_by_name(self.network, names)
            # param = getattr(module, attr).flatten()
            # print(param)
            # print('-'*50)

    @staticmethod
    def get_module_by_name(module, names):
        return reduce(getattr, names, module)