import torch
import random
import os
import gc
import time

def detect_gpus():
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible_devices:
        visible_gpus = [int(gpu.strip()) for gpu in cuda_visible_devices.split(',') if gpu.strip()]
        print(f"通过 CUDA_VISIBLE_DEVICES 检测到 {len(visible_gpus)} 张GPU: {visible_gpus}")
        return visible_gpus
    else:
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 张GPU")
        return list(range(num_gpus))

def matrix_multiplication_task(global_device_id, local_device_id):
    device = torch.device(f'cuda:{local_device_id}')
    matrix_size = 2048
    matrix1 = torch.randn(matrix_size, matrix_size, device=device)
    matrix2 = torch.randn(matrix_size, matrix_size, device=device)
    result = torch.matmul(matrix1, matrix2)
    return result

def cleanup_gpu_memory():
    """
    彻底清理GPU显存
    """
    # 清理PyTorch缓存
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU显存已清理")

def cal_mat():
    visible_gpus = detect_gpus()
    if not visible_gpus:
        print("未检测到GPU，退出")
        return
    
    iteration = 0
    while True:
        iteration += 1
        # print(f"--- iteration={iteration}\n")
        for local_index, global_id in enumerate(visible_gpus):
            matrix_multiplication_task(global_id, local_index)
        torch.cuda.synchronize()
        # if os.path.exists():
        #     cleanup_gpu_memory()
        #     print('cleaning memory')
        #     time.sleep(32)
        #     return

if __name__ == '__main__':
    cal_mat()

