DATASET_PATH=dataset/SemanticKITTI/dataset

for RATIO in 0.5 1 2 5 10 20 50; do
  # generate semi-supervised datalist
  # python semantickitti_01_make_list.py \
  #   --ratio $RATIO \
  #   --base_path $DATASET_PATH/sequences \
  #   --save_path $DATASET_PATH/

  # generate PLE-applied pseudo labels
  python semantickitti_02_ple.py \
    --ratio $RATIO \
    --base_path $DATASET_PATH \
    --save_path $DATASET_PATH/PLE_$RATIO

  # evaluate PLE-applied pseudo labels
  python semantickitti_03_evaluate.py \
    --gt $DATASET_PATH \
    --pred $DATASET_PATH/PLE_$RATIO

  # make pseudo labels list
  python semantickitti_04_make_pseudo_list.py \
    --ratio $RATIO \
    --base_path $DATASET_PATH \
    --save_path $DATASET_PATH \
    --pseudo_file_path $DATASET_PATH/PLE_$RATIO
done