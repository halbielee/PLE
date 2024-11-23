import os
import pickle
from argparse import ArgumentParser


def uniform_sampling(default_root, seq_list, ratio=0.5, save_path: str = "./"):
    """
    Perform uniform sampling on the given sequences and generate labeled and unlabeled data lists.

    Args:
        default_root (str): The root directory containing the sequence data.
        seq_list (list): List of sequence numbers to process.
        ratio (int, float, str, optional): The ratio of frames to sample as labeled data. Default is 0.1 (10%).
        save_path (str, optional): The directory to save the output pickle files. Default is './'.

    Returns:
        None

    This function reads the LiDAR and label files from the specified sequences, performs uniform sampling
    to select a subset of frames as labeled data, and treats the remaining frames as unlabeled data. The
    labeled and unlabeled data lists are saved as pickle files in the specified save_path.

    The labeled data includes paths to the LiDAR points and semantic mask files, while the unlabeled data
    includes only the paths to the LiDAR points. The function also prints the number of labeled and unlabeled
    frames, as well as the total number of frames processed.
    """

    labeled_dict = {"metainfo": {"DATASET": "SemanticKITTI"}, "data_list": []}
    unlabeled_dict = {"metainfo": {"DATASET": "SemanticKITTI"}, "data_list": []}

    total_frames = 0
    for seq_num in seq_list:
        files = os.listdir(os.path.join(default_root, seq_num, "velodyne"))
        files.sort()

        total_frames += len(files)
        # equally spaced sampling with 10% of the total frames
        num_frames = len(files)
        num_sample = int(num_frames * float(ratio))
        sample_interval = num_frames // num_sample

        labeled_files = []
        for i in range(num_sample):
            labeled_files.append(files[i * sample_interval])

        labeled_files.sort()
        unlabeled_files = list(set(files) - set(labeled_files))
        unlabeled_files.sort()

        for file in labeled_files:
            data_format = {
                "lidar_points": {"lidar_path": None, "num_pts_feats": 4},
                "pts_semantic_mask_path": None,
                "sample_id": None,
            }

            data_format["lidar_points"]["lidar_path"] = os.path.join(
                "sequences", seq_num, "velodyne", file
            )
            data_format["pts_semantic_mask_path"] = os.path.join(
                "sequences", seq_num, "labels", file
            )
            data_format["sample_id"] = str(int(seq_num)) + str(int(file.split(".")[0]))

            labeled_dict["data_list"].append(data_format)

        for file in unlabeled_files:
            data_format = {
                "lidar_points": {"lidar_path": None, "num_pts_feats": 4},
                "sample_id": None,
            }

            data_format["lidar_points"]["lidar_path"] = os.path.join(
                "sequences", seq_num, "velodyne", file
            )
            data_format["sample_id"] = str(int(seq_num)) + str(int(file.split(".")[0]))

            unlabeled_dict["data_list"].append(data_format)

    print("labeled_list: ", len(labeled_dict["data_list"]))
    print("unlabeled_list: ", len(unlabeled_dict["data_list"]))
    print("total_frames: ", total_frames)

    save_pkl_file(
        os.path.join(save_path, f"semantickitti_infos_train.{ratio}.pkl"), labeled_dict
    )
    save_pkl_file(
        os.path.join(save_path, f"semantickitti_infos_train.{ratio}-unlabeled.pkl"),
        unlabeled_dict,
    )

    print(
        "labeled is saved at: ",
        os.path.join(save_path, f"semantickitti_infos_train.{ratio}.pkl"),
    )
    print(
        "unlabeled is saved at: ",
        os.path.join(save_path, f"semantickitti_infos_train.{ratio}-unlabeled.pkl"),
    )


def check_file_format(file):
    labeled_pkl = pickle.load(open(os.path.join(file), "rb"))
    print(labeled_pkl.keys())
    print(labeled_pkl["metainfo"])
    print(labeled_pkl["data_list"][0])
    print(labeled_pkl["data_list"][5010])


def save_pkl_file(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def get_arguments():
    # Argument Parser
    parser = ArgumentParser("./semantickitti_01_make_list.py")
    parser.add_argument(
        "--base_path",
        type=str,
        default="~/dataset/SemanticKITTI/dataset/sequences",
        help="Path to the SemanticKITTI dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="~/dataset/SemanticKITTI/dataset/",
        help="Path to save the labeled and unlabeled data lists",
    )
    parser.add_argument(
        "--ratio", type=str, default=0.5, help="Ratio of labeled data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    train_sequence_list = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]

    uniform_sampling(
        args.base_path, train_sequence_list, ratio=args.ratio, save_path=args.save_path
    )
