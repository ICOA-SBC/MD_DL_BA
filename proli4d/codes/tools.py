import os
import time
import hostlist


class Distribution:
    # get SLURM variables
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    master_rank = (rank == 0)

    def __init__(self):
        # get node list from slurm
        hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

        # get IDs of reserved GPU
        gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
        # define MASTER_ADD & MASTER_PORT
        os.environ['MASTER_ADDR'] = hostnames[0]
        os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids)))  # to avoid port conflict on the same node

def convert_byte(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def master_print(text):
    if Distribution.master_rank:
        print(text)





