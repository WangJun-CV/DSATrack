# DSATrack

## Install the environment
Use the Anaconda (CUDA 11.0)
```
conda create -n DSATrack python=3.8
conda activate DSATrack
bash install.sh
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
The training datasets should look like this:
   ```
   -- lasot
        -- train
            |-- airplane
            |-- basketball
            |-- bear
            ...
   -- got10k
        |-- test
        |-- train
        |-- val
   -- coco
        |-- annotations
        |-- images
   -- trackingnet
        |-- TRAIN_0
        |-- TRAIN_1
        ...
        |-- TRAIN_11
        |-- TEST
   ```


## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python lib/train/run_training.py --script DSATrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/DSATrack`. 


## Evaluation

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py DSATrack vitb_256_mae_ce_32x4_ep300 --dataset lasot --threads 0 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py DSATrack vitb_256_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 0 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name DSATrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep100
```
- TrackingNet
```
python tracking/test.py DSATrack vitb_256_mae_ce_32x4_ep300 --dataset trackingnet --threads 0 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name DSATrack --cfg_name vitb_256_mae_ce_32x4_ep300
```

## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX3060 GPU.

```
# Profiling vitb_256_mae_ce_32x4_ep300
python tracking/profile_model.py --script DSATrack --config vitb_256_mae_ce_32x4_ep300
# Profiling vitb_256_mae_ce_32x4_ep300
python tracking/profile_model.py --script DSATrack --config vitb_256_mae_ce_32x4_ep300
```

