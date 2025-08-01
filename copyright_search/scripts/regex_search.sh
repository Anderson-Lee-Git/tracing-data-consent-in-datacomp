#!/bin/bash
#SBATCH --job-name=copyright-search
#SBATCH --account=
#SBATCH --partition=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-505
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --output=
#SBATCH --chdir=

BASE_DIR=
# Get shard directory from array task ID
shards=($(ls -d $BASE_DIR/* | sort))
SHARD_DIR=${shards[$SLURM_ARRAY_TASK_ID]}

# Get metadata file path
METADATA_FILE="$SHARD_DIR/ppocr_result_metadata.parquet"

# Run search script
python3 copyright_search.py --metadata-file $METADATA_FILE