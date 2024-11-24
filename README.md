# Learning from Spatio-temporal Correlation for Semi-Supervised LiDAR Semantic Segmentation

<p align="center"><a href="https://arxiv.org/abs/2410.06893">arXiv Paper</a></p>

<p align="center">Seungho Lee<sup>†</sup>, Hwijeong Lee<sup>‡</sup>, Hyunjung Shim<sup>‡</sup></p>
<p align="center"><sup>†</sup> <sub>Yonsei University</sub>, <sup>‡</sup> <sub>Korea Advanced Institute of Science and Technology</sub></p>


## Introduction
This novel semi-supervised LiDAR segmentation method leverages spatio-temporal information between adjacent scans to generate high-quality pseudo-labels, achieving state-of-the-art performance on SemanticKITTI and nuScenes with minimal labeled data (as low as 5%). Notably, it outperforms previous SOTA results using only 20% of labeled data, making it highly efficient for real-world applications.

## Dataset Preparation
To run semi-supervised LiDAR segmentation (SSLS), you'll need to download and preprocess the [SemanticKITTI](http://semantic-kitti.org/) and [nuScenes](https://www.nuscenes.org/) datasets. For detailed instructions on dataset preparation, please refer to our guide [here📚](docs/DATA_PREPARE.md).

## Installation
Please refer to the [installation guide](docs/INSTALL.md) for detailed instructions on setting up the environment.
This code is slightly modified from the original MM3D repository for dynamically loading the path of the dataset.
For the original MM3D repository, please refer to [here](https://github.com/open-mmlab/mmdetection3d).

## Run: Proximity-based Label Estimation (PLE)

### Option 1: Run the entire process
This repository includes an implementation of proximity-based label estimation (PLE).
You can run the entire process using the following commands (note that processing all labeled ratios (0.5, 1, 2, 5, 10, 20, 50) will take several hours):
```
cd generate_ple
bash semantickitti.sh
```

For step-by-step implementation, follow these instructions:
1. Set your environment variables:
   ```
   DATASET_PATH=~/dataset/SemanticKITTI/dataset
   RATIO=0.5    # 0.5, 1, 2, 5, 10, 20, or 50.  
   ```
2. Generate PLE-based pseudo labels:
   ```
   python semantickitti_02_ple.py \
   --ratio $RATIO \
   --base_path $DATASET_PATH \
   --save_path $DATASET_PATH/PLE_$RATIO
   ```

3. Evaluate the generated pseudo labels:
   ```
   python semantickitti_03_evaluate.py \
   --gt $DATASET_PATH \
   --pred $DATASET_PATH/PLE_$RATIO
   ```
4. Create a list of pseudo labels:
   ```
   python semantickitti_04_make_pseudo_list.py \
   --ratio $RATIO \
   --base_path $DATASET_PATH \
   --save_path $DATASET_PATH \
   --pseudo_file_path $DATASET_PATH/PLE_$RATIO
   ```

For nuScenes dataset, please refer to `nuscenes.sh` in the same directory.

### Option 2: Use pre-generated pseudo labels
If you want to use pre-generated pseudo labels, download the pseudo labels from the following links:
- [IROS2024_PLE](https://drive.google.com/drive/folders/1QdbhRVklB2abHPrcMAz1lwZXJ_iAELxD?usp=drive_link)

After downloading the pseudo labels, place them in the following directories:
- SemanticKITTI: 
  - `~/dataset/SemanticKITTI/dataset/PLE_$RATIO`
  - `~/dataset/SemanticKITTI/dataset/semantickitti_infos_train.ple.${RATIO}.pkl`
  - `~/dataset/SemanticKITTI/dataset/semantickitti_infos_train.ple.${RATIO}-unlabeled.pkl`
- nuScenes: 
  - `~/dataset/nuScenes/PLE_$RATIO`
  - `~/dataset/nuScenes/nuscenes_kitti_infos_train.ple.${RATIO}.pkl`
  - `~/dataset/nuScenes/nuscenes_kitti_infos_train.ple.${RATIO}-unlabeled.pkl`


## Run: Training Dual-branch Network
Execute the following script to train the dual-branch network with PLE-based pseudo labels:
```bash
bash script/lasermix_cy3d_mt_dualbranch_semi_semantickitti_ple.sh
```
You can also train the MeanTeacher model or use the nuScenes dataset. For more details, refer to the `script` directory.  


## Results

### Performance of mIoU on SemanticKITTI Dataset
| Method            | 0.5% | 1%  | 2%  | 5%  | 10% | 20% | 50% |
|-------------------|------|-----|-----|-----|-----|-----|-----|
| LaserMix          | 47.3 | 55.5| 59.2| 61.7| 62.4| 62.4| 62.1|
| PLE + Dual Branch | 52.2 | 61.1| 62.9| 62.8| 63.1| 64.1| 64.3|


### Performance of mIoU on nuScenes Dataset
| Method            | 0.5% | 1%  | 2%  | 5%  | 10% | 20% | 50% |
|-------------------|------|-----|-----|-----|-----|-----|-----|
| LaserMix          | 51.4 | 58.4| 63.9| 69.7| 71.6| 73.7| 73.7|
| PLE + Dual Branch | 58.0 | 62.9| 67.2| 72.8| 74.3| 76.0| 76.1|

Please See the [paper](https://arxiv.org/abs/2410.06893) for more details.

## Citation
If you find our work useful in your research, please cite:
```bibtex
@article{lee2023learning,
   title={Learning from Spatio-temporal Correlation for Semi-Supervised LiDAR Semantic Segmentation},
   author={Lee, Seungho and Lee, Hwijeong and Shim, Hyunjung},
   journal={arXiv preprint arXiv:2308.12345},
   year={2023}
}
```

## Acknowledgements
This code is hardly based on the [LaserMix](https://github.com/ldkong1205/LaserMix). 

