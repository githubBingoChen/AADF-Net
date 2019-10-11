# coding: utf-8
import os

datasets_root = '/home/b3-542/Documents/datasets/SaliencyDatasets'

# For each dataset, I put images and masks together
msra10k_path = os.path.join(datasets_root, 'msra10k')
ecssd_path = os.path.join(datasets_root, 'ecssd')
hkuis_path = os.path.join(datasets_root, 'hkuis')
pascals_path = os.path.join(datasets_root, 'pascals')
dutomron_path = os.path.join(datasets_root, 'dutomron')
duts_path = os.path.join(datasets_root, 'duts')
duts_train_path = os.path.join(datasets_root, 'duts_train')
sod_path = os.path.join(datasets_root, 'sod')
soc_path = os.path.join(datasets_root, 'soc')
soc_val_path = os.path.join(datasets_root, 'soc_val')
thur15k_path = os.path.join(datasets_root, 'thur15k')

pytorch_pretrained_root = '/home/b3-542/Packages/Models/PyTorch Pretrained'
pretrained_res50_path = os.path.join(pytorch_pretrained_root, 'ResNet', 'resnet50-19c8e357.pth')
