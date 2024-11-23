DATASET_PATH=dataset/nuscenes_kitti/

for RATIO in 0.5 1 2 5 10 20 50;do
  # generate semi-supervised datalist
  # python nuscenes_01_make_list.py \
  # --base_path $DATASET_PATH \
  # --save_path $DATASET_PATH \
  # --ratio $RATIO

  # generate PLE-applied pseudo labels
  python nuscenes_02_ple.py \
    --ratio $RATIO \
    --base_path $DATASET_PATH \
    --save_path $DATASET_PATH/PLE_$RATIO

  # evaluate PLE-applied pseudo labels
  python nuscenes_03_evaluate.py \
    --gt $DATASET_PATH \
    --pred $DATASET_PATH/PLE_$RATIO

  # make pseudo labels list
  python nuscenes_04_make_pseudo_list.py \
    --base_path $DATASET_PATH \
    --save_path $DATASET_PATH \
    --ratio $RATIO \
    --pseudo_file_path $DATASET_PATH/PLE_$RATIO
done