from random import shuffle
import torch
from torch.distributed import rpc
import random
from tools import *
from .tools import weighted_allocate

def network_synchronize_withmask(remote_networks, network, gpus, process_per_gpu):
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    mask_dict = {
        name: p for name, p in network.named_buffers() if 'mask' in name
    }
    params_to_be_updated = []
    for name_id, (key, param) in enumerate(params_dict.items()):
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()
        param_flat = param.flatten()
        indices = mask_flat.nonzero().flatten().tolist()
        params_to_be_updated += [[name_id, idx, param_flat[idx].item()] for idx in indices]
    params_to_be_updated_rref = rpc.RRef(torch.Tensor(params_to_be_updated))
    for gpu in gpus:
        for i in range(process_per_gpu):
            remote_networks[f"{gpu}-{i}"].rpc_async().synchronize(params_to_be_updated_rref)
            
def network_synchronize(remote_networks, network, gpus, process_per_gpu):
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    params_to_be_updated = []
    for name_id, (key, param) in enumerate(params_dict.items()):
        #改成所有flatten
        params_to_be_updated.append(param.cpu().flatten())
    params_to_be_updated = torch.cat(params_to_be_updated)
    params_to_be_updated_rref = rpc.RRef(torch.Tensor(params_to_be_updated))
    for gpu in gpus:
        for i in range(process_per_gpu):
            remote_networks[f"{gpu}-{i}"].rpc_async().synchronize(params_to_be_updated_rref)

def cge_weight_allocate_to_process_withmask(remote_networks, network, gpus, process_per_gpu, param_name_to_module_id, time_consumption_per_layer):
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    mask_dict = {
        name: p for name, p in network.named_buffers() if 'mask' in name
    }
    whole_size = len(gpus) * process_per_gpu
    params_to_be_perturbed = []
    time_consumption = []
    for name_id, (key, param) in enumerate(params_dict.items()):
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()
        indices = mask_flat.nonzero().flatten().tolist()
        params_to_be_perturbed += [(name_id, idx) for idx in indices]
        time_consumption += [time_consumption_per_layer[param_name_to_module_id(key)] for _ in indices]
    params_to_be_perturbed = weighted_allocate(params_to_be_perturbed, time_consumption, whole_size)
    params_set_signal = []
    param_names = list(params_dict.keys())
    for j, gpu in enumerate(gpus):
        for i in range(process_per_gpu):
            idx = (j * process_per_gpu) + i
            params_to_be_perturbed[idx] = rpc.RRef(torch.Tensor(params_to_be_perturbed[idx]))
            params_set_signal.append(remote_networks[f"{gpu}-{i}"].rpc_async().set_params_to_be_perturbed(params_to_be_perturbed[idx], param_names))
    for pss in params_set_signal:
        pss.wait()


def cge_weight_allocate_to_process(remote_networks, network, gpus, process_per_gpu, param_name_to_module_id):
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    whole_size = len(gpus) * process_per_gpu
    params_to_be_perturbed = []    
    total_pertube_num = param_name_to_module_id(name='depth')
    for cuurent_pertube_id in range(total_pertube_num):
        params_to_be_perturbed.append(cuurent_pertube_id)
    params_to_be_perturbed = weighted_allocate(params_to_be_perturbed, whole_size)
    param_names = {}
    for name_id, key in enumerate(params_dict.keys()):
        cuurent_pertube_id = param_name_to_module_id(key)
        if cuurent_pertube_id not in param_names.keys():
            param_names[cuurent_pertube_id] = [key]
        else:
            param_names[cuurent_pertube_id].append(key)
    param_names = [value for value in param_names.values()]  #param_names每一个block对应哪些需要扰动的参数层
    params_set_signal = []
    for j, gpu in enumerate(gpus):
        for i in range(process_per_gpu):
            idx = (j * process_per_gpu) + i
            params_to_be_perturbed[idx] = rpc.RRef(torch.Tensor(params_to_be_perturbed[idx]))  #params_to_be_perturbed->[[0,1,2],...]每一个设备需要扰动哪些block
            params_set_signal.append(remote_networks[f"{gpu}-{i}"].rpc_async().set_params_to_be_perturbed(params_to_be_perturbed[idx], param_names))
    for pss in params_set_signal:
        pss.wait()
    return params_to_be_perturbed, param_names

def cge_calculation(remote_networks, network, gpus, process_per_gpu, x, y, cge_step_size, params_to_be_perturbed, param_names):
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    device = next(network.parameters()).device
    random_seeds = []
    x_rref, y_rref = rpc.RRef(x), rpc.RRef(y)
    grads_signal = []
    for gpu in gpus:
        for i in range(process_per_gpu):
            random_seed = random.randint(0,10000)
            random_seeds.append(random_seed)
            grads_signal.append(remote_networks[f"{gpu}-{i}"].rpc_async(timeout=0).calculate_grads(x_rref, y_rref, cge_step_size, random_seed))
    grads = []
    for g in grads_signal:
        grads.append(g.wait())
    grads = torch.cat(grads, dim=0).to(device)
    grads_dict = {}
    for i in range(len(grads)):
        grads_dict[int(grads[i, 0])] = grads[i, 1]
    # print(grads_dict)
    from models.vit import param_name_to_module_id_vit
    param_name_to_module_id = param_name_to_module_id_vit
    
    cnt_ls = []
    cnt = 0
    for i in range(len(params_to_be_perturbed)):
        cnt += sum([len(param_names[int(j)]) for j in params_to_be_perturbed[i]])
        cnt_ls.append(cnt)
    # print('params_to_be_perturbed: {}'.format(params_to_be_perturbed))
    # print('param_names: {}'.format(param_names))
    # print('cnt_ls: {}'.format(cnt_ls))
    
    current_device_id = 0
    set_seed(random_seeds[current_device_id])
    for name_id, (key, param) in enumerate(params_dict.items()):
        if name_id == cnt_ls[current_device_id]:
            current_device_id += 1
            set_seed(random_seeds[current_device_id])
        pertube_id = param_name_to_module_id(key)
        perturb = torch.randn_like(param)
        perturb /= (torch.norm(perturb) + 1e-8)
        perturb *= cge_step_size
        # print("key: {}, random seed11: {}".format(key, perturb))
        param.grad = perturb * grads_dict[pertube_id]   
             
        # param.grad = torch.zeros_like(param)
        # grads_indices_and_values = grads[grads[:, 0]==name_id, 1:]
        # param_grad_flat = param.grad.flatten()
        # param_grad_flat[grads_indices_and_values[:, 0].long()] = grads_indices_and_values[:, 1]

