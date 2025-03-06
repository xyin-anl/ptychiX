import subprocess
import re

def select_gpu(params):
    # Get GPU stats
    gpu_stats = get_gpu_usage()

    if gpu_stats is None or len(gpu_stats) == 0:
        # Default to GPU 0 if can't get stats
        return 0
    else:
        # Find GPU with 0% utilization and lowest memory usage
        zero_util_gpus = [gpu for gpu in gpu_stats if gpu['utilization'] == 0]
        
        if zero_util_gpus:
            # Among GPUs with 0% utilization, pick the one with lowest memory usage
            selected_gpu = min(zero_util_gpus, key=lambda x: x['memory_percent'])
        else:
            print(f"No GPU has 0% utilization. Selecting GPU with lowest combined score.")
            # If no GPU has 0% utilization, pick the one with lowest combined score
            selected_gpu = min(gpu_stats, 
                                key=lambda x: x['utilization'] + x['memory_percent'])
        
        return selected_gpu['index']

# Check GPU usage and select least utilized GPU
def get_gpu_usage():
    try:
        # Run nvidia-smi to get GPU utilization and memory usage
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8').strip().split('\n')
        
        gpu_stats = []
        for line in output:
            index, util, mem_used, mem_total = line.split(', ')
            gpu_stats.append({
                'index': int(index),
                'utilization': int(util),
                'memory_used': int(mem_used),
                'memory_total': int(mem_total),
                'memory_percent': (int(mem_used) / int(mem_total)) * 100
            })
        return gpu_stats
    except:
        return None
