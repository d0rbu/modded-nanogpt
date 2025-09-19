#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=modded-nanogpt       # Set the job name to "get_activations"
#SBATCH --time=24:00:00                 # Set the wall clock limit to 6 hours
#SBATCH --ntasks=1                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --output=modded-nanogpt.%j      # Send stdout/err to "modded-nanogpt.[jobID]"
#SBATCH --error=modded-nanogpt.%j.err   # Send stderr to separate file
#SBATCH --gres=gpu:a100:1               # Request 2 "a100" GPUs per node
#SBATCH --partition=gpu                 # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                # Set billing account to 123456
##SBATCH --mail-type=ALL                 # Send email on all job events
##SBATCH --mail-user=email_address       # Send all emails to email_address

# Enable detailed logging
set -x

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# Print SLURM environment information for debugging
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM Number of Nodes: $SLURM_NNODES"
echo "SLURM Number of Tasks: $SLURM_NTASKS"
echo "SLURM Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "SLURM Local ID: $SLURM_LOCALID"
echo "SLURM Procedure ID: $SLURM_PROCID"
echo "SLURM Node ID: $SLURM_NODEID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to the project directory
cd ~/modded-nanogpt

# Run the distributed job
run.sh
