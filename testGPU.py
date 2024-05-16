
import torch

def try_gpu(i=0):
  '''如果可用，返回第 i 个 GPU 的设备对象（i 默认为 0）；否则返回 CPU 的设备对象'''
  if torch.cuda.device_count() >= i+1:
      return torch.device(f'cuda:{i}')
  return torch.device('cpu')
def try_all_gpus():
  '''返回所有可用 GPU 的设备对象列表；否则返回 CPU 的设备对象（也是列表） '''
  devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
  return devices if devices else [torch.device('cpu')]

