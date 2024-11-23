import os
import pickle
import os.path as osp
import copy


datainfo = {
    "metainfo": {"DATASET": "SemanticKITTI"},
    "data_list": {
        "lidar_points": {
            "lidar_path": "sequences/00/velodyne/000002.bin",
            "num_pts_feats": 4,
        },
        "pts_semantic_mask_path": "sequences/00/labels/000002.label",
        "sample_id": "02",
    },
}


class DataGenerator:
    """
    A class to generate labeled and unlabeled data lists for the SemanticKITTI dataset,
    including pseudo-labeled data.
    Attributes:
        ratio (str): The ratio of pseudo-labeled data to be included.
        base_path (str): The base path where the labeled and unlabeled data files are located.
        save_path (str): The path where the generated data files will be saved.
        labeled_name (str): The name of the labeled data file.
        unlabeled_name (str): The name of the unlabeled data file.
        pseudo_file_path (str): The path where the pseudo-labeled data files are located.
        pseudo_folder (str): The folder name extracted from the pseudo_file_path.
    Methods:
        generate_info():
            Generates the labeled and unlabeled data lists, including pseudo-labeled data,
            and saves them to the specified save_path.
    """

    def __init__(
        self,
        ratio,
        base_path,
        save_path,
        labeled_name,
        unlabeled_name,
        pseudo_file_path,
    ):
        self.ratio = ratio
        self.base_path = base_path
        self.save_path = save_path
        self.labeled_name = labeled_name
        self.unlabeled_name = unlabeled_name
        self.pseudo_file_path = pseudo_file_path
        self.pseudo_folder = pseudo_file_path.split("/")[-1]

    def generate_info(self):
        save_dict = {"metainfo": {"DATASET": "SemanticKITTI"}, "data_list": []}
        # load labeled & unlabeled list file
        labeled_pkl = pickle.load(
            open(os.path.join(self.base_path, self.labeled_name), "rb")
        )
        unlabeled_pkl = pickle.load(
            open(os.path.join(self.base_path, self.unlabeled_name), "rb")
        )

        labeled_datalist = labeled_pkl["data_list"]
        unlabeled_datalist = unlabeled_pkl["data_list"]

        # extract lidar_id from labeled & unlabeled datalist
        labeled_lidar_id = []
        unlabeled_lidar_id = []
        for item in labeled_datalist:
            labeled_lidar_id.append(item["lidar_points"]["lidar_path"])

        for item in unlabeled_datalist:
            unlabeled_lidar_id.append(item["lidar_points"]["lidar_path"])

        # get pseudo_file list
        sequences = sorted(os.listdir(self.pseudo_file_path))
        pseudo_file_list = []
        for seq in sequences:
            # print(seq)
            label_files = os.listdir(os.path.join(self.pseudo_file_path, seq, "labels"))
            for label_file in label_files:
                lidar_path = (
                    f'sequences/{seq}/velodyne/{label_file.replace(".label", ".bin")}'
                )
                pseudo_file_list.append(lidar_path)

        # add pseudo_file_list to labeled_datalist and delete from unlabeled_datalist
        new_labeled_datalist = list(set(labeled_lidar_id + pseudo_file_list))
        new_unlabeled_datalist = list(set(unlabeled_lidar_id) - set(pseudo_file_list))

        # print each length of datalist (labeled & unlabeled, new_labeled & new_unlabeled)
        print("The number of labeled & unlabeled data")
        print("labeled: ", len(labeled_datalist))
        print("unlabeled: ", len(unlabeled_datalist))
        print("The number of new labeled & new unlabeled data")
        print("new_labeled: ", len(new_labeled_datalist))
        print("new_unlabeled: ", len(new_unlabeled_datalist))

        # sort each new_dataslist
        new_labeled_datalist.sort()
        new_unlabeled_datalist.sort()

        # make an instance of save_dict
        labeled_dict = copy.deepcopy(save_dict)
        for lidar_path in new_labeled_datalist:
            if lidar_path in labeled_lidar_id:
                data = {
                    "lidar_points": {"lidar_path": lidar_path, "num_pts_feats": 4},
                    "pts_semantic_mask_path": lidar_path.replace(
                        "velodyne", "labels"
                    ).replace(".bin", ".label"),
                    "sample_id": lidar_path.split("/")[-1].split(".")[0],
                }
            elif lidar_path in pseudo_file_list:
                data = {
                    "lidar_points": {"lidar_path": lidar_path, "num_pts_feats": 4},
                    # 'pts_semantic_mask_path': lidar_path.replace('sequences', f'pseudo_{self.ratio}').replace('velodyne', 'labels').replace('.bin', '.label'),
                    "pts_semantic_mask_path": f'{self.pseudo_folder}/{lidar_path.split("/")[1]}/labels/{lidar_path.split("/")[-1].replace(".bin", ".label")}',
                    "sample_id": lidar_path.split("/")[-1].split(".")[0],
                }
            else:
                raise ValueError(
                    "lidar_path is not in labeled_lidar_id or pseudo_file_list"
                )
            labeled_dict["data_list"].append(data)
        
        # deepcopy unlabeled_dict
        unlabeled_dict = copy.deepcopy(save_dict)
        for lidar_path in new_unlabeled_datalist:
            data = {
                "lidar_points": {"lidar_path": lidar_path, "num_pts_feats": 4},
                "pts_semantic_mask_path": lidar_path.replace(
                    "velodyne", "labels"
                ).replace(".bin", ".label"),
                "sample_id": lidar_path.split("/")[-1].split(".")[0],
            }
            unlabeled_dict["data_list"].append(data)

        # save labeled_dict & unlabeled_dict
        with open(
            osp.join(self.save_path, f"semantickitti_infos_train.ple.{self.ratio}.pkl"),
            "wb",
        ) as f:
            pickle.dump(labeled_dict, f)

        with open(
            osp.join(
                self.save_path,
                f"semantickitti_infos_train.ple.{self.ratio}-unlabeled.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(unlabeled_dict, f)
            
        # print the path of saved pseudo data (with its name)
        print("\nThe pseudo data is saved at:")
        print(
            "Labeled data: ",
            osp.join(self.save_path, f"semantickitti_infos_train.ple.{self.ratio}.pkl"),
        )
        print(
            "Unlabeled data: ",
            osp.join(
                self.save_path,
                f"semantickitti_infos_train.ple.{self.ratio}-unlabeled.pkl",
            ),
        )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo list")
    parser.add_argument(
        "--base_path",
        type=str,
        default="~/dataset/SemanticKITTI/dataset/",
        help="base path of dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="~/dataset/SemanticKITTI/dataset/",
        help="save path of pseudo",
    )
    parser.add_argument("--ratio", type=str, default="0.5", help="ratio of pseudo data")
    parser.add_argument(
        "--labeled_name",
        type=str,
        default="semantickitti_infos_train.{}.pkl",
        help="name of labeled data",
    )
    parser.add_argument(
        "--unlabeled_name",
        type=str,
        default="semantickitti_infos_train.{}-unlabeled.pkl",
        help="name of unlabeled data",
    )
    parser.add_argument(
        "--pseudo_file_path", type=str, default="R0.5_PLE", help="path of pseudo data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_generator = DataGenerator(
        ratio=args.ratio,
        base_path=args.base_path,
        save_path=args.save_path,
        labeled_name=args.labeled_name.format(args.ratio),
        unlabeled_name=args.unlabeled_name.format(args.ratio),
        pseudo_file_path=args.pseudo_file_path,
    )
    data_generator.generate_info()
