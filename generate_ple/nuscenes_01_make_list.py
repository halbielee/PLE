import pickle
import os.path as osp
import numpy as np

label_mapping = {
    1: 0,
    5: 0,
    7: 0,
    8: 0,
    10: 0,
    11: 0,
    13: 0,
    19: 0,
    20: 0,
    0: 0,
    29: 0,
    31: 0,
    9: 1,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    21: 6,
    2: 7,
    3: 7,
    4: 7,
    6: 7,
    12: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    30: 16,
}


def make_new_info_files(
    base_root,
    save_root,
    train_file,
    ratio,
):
    train_data = np.load(osp.join(base_root, train_file), allow_pickle=True)

    train_datalist = train_data["data_list"]
    num_train_data = len(train_datalist)
    print("The number of train data", num_train_data)
    dst_num_labeled = int(num_train_data * float(ratio) * 0.01)
    dst_num_unlabeled = num_train_data - dst_num_labeled

    # select labeled data from new_datalist
    labeled_idx = np.random.choice(num_train_data, dst_num_labeled, replace=False)
    labeled_idx.sort()
    labeled_datalist = [train_datalist[i] for i in labeled_idx]

    # select unlabeled data from new_datalist
    unlabeled_idx = np.setdiff1d(np.arange(num_train_data), labeled_idx)
    unlabeled_idx.sort()
    unlabeled_datalist = [train_datalist[i] for i in unlabeled_idx]

    # delete pts_semantic_mask_path from unlabeled_datalist
    for i in range(len(unlabeled_datalist)):
        del unlabeled_datalist[i]["pts_semantic_mask_path"]

    print(
        "Labeled: {}, Unlabeled: {}".format(
            len(labeled_datalist), len(unlabeled_datalist)
        )
    )

    # make new labeled_data
    new_labeled_data = train_data.copy()
    new_labeled_data["data_list"] = labeled_datalist
    new_labeled_name = train_file.replace(".pkl", f".{ratio}.pkl")

    # make new unlabeled_data
    new_unlabeled_data = train_data.copy()
    new_unlabeled_data["data_list"] = unlabeled_datalist
    new_unlabeled_name = train_file.replace(".pkl", f".{ratio}-unlabeled.pkl")

    print("new_labeled_name", new_labeled_name)
    print("new_unlabeled_name", new_unlabeled_name)

    # save new labeled_data
    with open(osp.join(save_root, new_labeled_name), "wb") as f:
        pickle.dump(new_labeled_data, f)

    # save new unlabeled_data
    with open(osp.join(save_root, new_unlabeled_name), "wb") as f:
        pickle.dump(new_unlabeled_data, f)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate file list for nuScenes dataset"
    )
    parser.add_argument("--base_path", type=str, default="", help="base path")
    parser.add_argument("--save_path", type=str, default="", help="save path")
    parser.add_argument(
        "--train_name",
        type=str,
        default="nuscenes_kitti_infos_train.pkl",
        help="train_name",
    )
    parser.add_argument("--ratio", type=str, default="0.5", help="ratio")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    make_new_info_files(args.base_path, args.save_path, args.train_name, args.ratio)
