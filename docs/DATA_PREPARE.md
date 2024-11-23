# Data Preparation

### Directory Structure
The ideal dataset directory structure for this code shoudl be:
```
└── dataset 
    │── SemanticKITTI # customizable path
    └── nuscenes_kitti # customizable path
```

### SemanticKITTI Dataset Setup
You need to download the official "SemanticKITTI" dataset.

#### Download Instructions
1. Download the [SemanticKITTI](http://semantic-kitti.org/) dataset files: 
    ```
    data_odometry_calib.zip # Calibration data
    data_odometry_labels.zip # lLbel data
    data_odometry_velodyne.zip # 3d point cloud data
    ```
2. Extract the files into your `dataset/SemanticKITTI` directory:
    ```
    unzip data_odometry_calib.zip -d dataset/SemanticKITTI/
    unzip data_odometry_labels.zip -d dataset/SemanticKITTI/
    unzip data_odometry_velodyne.zip -d dataset/SemanticKITTI/
    ```
3. After extraction, your directory structure should look like this:
    ```
        └── SemanticKITTI
            └── dataset
                └── sequences
                    │── 00
                         ├── velodyne   <-.bin files (3d pointcloud)
                         │── labels     <-.label files 
                         │── calib.txt 
                         │── poses.txt
                         └── times.txt
                    │── 01
                    │──  ·
                    │──  ·
                    │──  ·
                    └── 21
                    └── semantic-kitti.yaml
    ```

#### Semi-supervised Learning Setup
For semi-supervised LiDAR segmentation (SSLS), you'll need preprocessed data split files for training and validation:
- Download our prepared .pkl files (datalist and pseudo-labels) from [here](https://drive.google.com/drive/folders/19aRqsXh7BYZ4LxxoW8udY8S9DdfdMtvz?usp=drive_link), or
- Generate them yourself using `generate_ple/semantickitti_01_make_list.py`

--

### nuScenes Dataset Setup
You need to download the official nuScenes dataset and process the dataset into `SemanticKITTI` style. 

#### Download Instructions
1. Download the [nuScenes-lidarseg](https://www.nuscenes.org/nuscenes) dataset files:
    ```
    - v1.0-trainval{01-10}_blobs.tar
    - nuScenes-lidarseg-all-v1.0.tar.bz2
    - v1.0-trainval_meta.tgz
    ```
2. Extract the files into your `dataset` directory:
    ```
    # Extract point cloud data 
    for VAR in 03 04 05 06 07 08 09 10
    do 
        tar -xzf "v1.0-trainval${VAR}_blobs.tgz -C dataset/nuscenes_org/"
    done

    # Extract metadata and labels
    tar -xzf v1.0-trainval_meta.tgz -C dataset/nuscenes_org/
    tar -xvf nuScenes-lidarseg-all-v1.0.tar.bz2 -C dataset/nuscenes_org/
    ```

    After extraction, your directory structure should look like this:
    ```
        └── nuscenes_org
            ├── Standard nuScenes folders (samples, sweep)
            │
            ├── lidarseg
            │   └── v1.0-{mini, test, trainval}  # Contains .bin files with point cloud
            │                                     # label data (note: v1.0-test has no
            │                                     # .bin files)
            │
            └── v1.0-{mini, test, trainval}
                ├── Standard files (attribute.json, calibrated_sensor.json, etc.)
                ├── lidarseg.json  # Maps .bin files to tokens   
                └── category.json  # Label category definitions (overwrites
                                  # nuScenes v1.0 category.json)
    ```
3. Convert the nuScenes dataset to SemanticKITTI format:
    - Clone the [nuscenes2kitti](https://github.com/PRBonn/nuscenes2kitti) repository
    - Run `nuscenes2kitti.py` (you can skip panoptic data-related code):
    ```
    mkdir dataset/nuscenes_kitti/sequences/

    cd nuscenes2kitti
    python nuscenes2kitti.py --nuscenes_dir $YOUR_PATH/dataset/nuscenes_org/ --output_dir  $YOUR_PATH/dataset/nuscenes_kitti/sequences/
    ```
4. The converted dataset structure should look like this:
    ```
    └── nuscenes_kitti
        ├── 0001
             ├── velodyne   # .bin files (3D point cloud)
             │── labels     # .label files 
             │── calib.txt 
             │── poses.txt
             │── lidar_tokens.txt
             └── files_mapping.txt
        │──   ·
        │──   ·
        │──   ·
        └── 1110
    ```

#### Semi-supervised Learning Setup
For semi-supervised LiDAR segmentation (SSLS), you'll need preprocessed data split files for training and validation:
- Download our prepared .pkl files (datalist and pseudo-labels) from [here](https://drive.google.com/drive/folders/19aRqsXh7BYZ4LxxoW8udY8S9DdfdMtvz?usp=drive_link), or
- Generate them yourself using `generate_ple/nuscenes_01_make_list.py`


---

### Citations

If you use these datasets, please cite the following papers:

#### nuScenes Dataset
```bibtex
@article{fong2022panopticnuscenes,
    author = {W. K. Fong and R. Mohan and J. V. Hurtado and L. Zhou and H. Caesar and O. Beijbom and A. Valada},
    title = {Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
    journal = {IEEE Robotics and Automation Letters},
    volume = {7},
    number = {2},
    pages = {3795--3802},
    year = {2022}
}
```
```bibtex
@inproceedings{caesar2020nuscenes,
    author = {H. Caesar and V. Bankiti and A. H. Lang and S. Vora and V. E. Liong and Q. Xu and A. Krishnan and Y. Pan and G. Baldan and O. Beijbom},
    title = {nuScenes: A Multimodal Dataset for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {11621--11631},
    year = {2020}
}
```

#### SemanticKITTI Dataset

```bibtex
@inproceedings{behley2019semantickitti,
    author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
    title = {SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages = {9297--9307},
    year = {2019}
}
```
```bibtex
@inproceedings{geiger2012kitti,
    author = {A. Geiger and P. Lenz and R. Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {3354--3361},
    year = {2012}
}
```