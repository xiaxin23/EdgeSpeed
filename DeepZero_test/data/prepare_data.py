import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import sys
sys.path.append(".")

def preload_to_cuda(dataset,device):
    data_list = []
    targets_list = []
    for img, target in tqdm(dataset):
        data_list.append(img.unsqueeze(0))  # 添加 batch 维度 [1, C, H, W]
        targets_list.append(target)
    # 合并为张量并移动到 CUDA
    data_tensor = torch.cat(data_list, dim=0).to(device) # [50000, 3, 32, 32]
    targets_tensor = torch.tensor(targets_list).to(device)  # [50000]
    return data_tensor, targets_tensor

class CUDADataset(Dataset):
    def __init__(self, data, device):
        # 直接将数据和标签一次性加载到 GPU
        self.data = data[0].to(device)
        self.targets = data[1].to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回 GPU 上的数据（无需每次传输）
        return self.data[idx], self.targets[idx]
    
def prepare_dataset(dataset, batch_size=128, pin_memory=True, device=None):
    from cfg import data_path
    data_path = os.path.join(data_path, dataset)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    if dataset == "mnist":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = train_transform
        train_data = datasets.MNIST(root = data_path, train = True, download = True, transform = train_transform)
        train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=2)
        test_data = datasets.MNIST(root = data_path, train = False, download = True, transform = test_transform)
        test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers=2)
        cls_num = 10
    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        test_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = train_transform)
        # data_tensor, targets_tensor = preload_to_cuda(train_data,device)
        # train_data = CUDADataset(train_data,device)
        train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=2, pin_memory=pin_memory)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = test_transform)
        # data_tensor, targets_tensor = preload_to_cuda(test_data,device)
        # test_data = CUDADataset(test_data,device)
        test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers=2, pin_memory=pin_memory)
        cls_num = 10
    else:
        raise NotImplementedError
    return {
        'train': train_loader,
        'test': test_loader,
    }, cls_num
