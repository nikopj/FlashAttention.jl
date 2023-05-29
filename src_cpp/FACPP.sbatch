#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=48                # uses 1 compute core per task
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=FlashAttantionCPP
#SBATCH --output=FlashAttantionCPPl1.out
#SBATCH --mail-type=END
#SBATCH --mail-user=lg3172@nyu.edu


module purge
module load intel/19.1.2 openmpi/intel/4.0.5 eigen/3.4.0 cuda/11.6.2

lscpu
make && ./FlashAttention