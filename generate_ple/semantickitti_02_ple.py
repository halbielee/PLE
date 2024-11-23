import os
import pickle
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
from scipy.spatial import KDTree
from collections import OrderedDict
from collections import deque
from argparse import ArgumentParser


def parse_calibration(filename):
    """read calibration file with given filename
    Returns
    -------
    dict
        Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """read poses file with per-scan poses from given filename
    Returns
    -------
    list
        list of poses as 4x4 numpy arrays.
    """
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)
    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def get_seq(lidar_path):
    return lidar_path.split("/")[1]


def get_frame(lidar_path):
    return int(lidar_path.split("/")[-1].split(".")[0])


def get_lidar_path(seq, frame):
    return os.path.join("sequences", seq, "velodyne", f"{frame:06d}.bin")


class ProgressivePLE:
    """
    Progressive Pseudo-Labeling Enhancement (PLE) class for processing LiDAR data.
    This class implements a progressive pseudo-labeling enhancement technique for LiDAR data.
    It identifies unlabeled LiDAR frames and assigns pseudo-labels based on the nearest labeled frames.
    Attributes:
        ratio (str): Ratio of labeled to unlabeled data. This will be used as float value.
        base_path (str): Base path for the dataset.
        save_path (str): Path to save the processed data.
        labeled_name (str): Template name for labeled data files.
        unlabeled_name (str): Template name for unlabeled data files.
        search_range (int): Range to search for nearest labeled frames.
        gt_labeled_list (list): List of ground truth labeled frames.
        pseudo_labeled_list (list): List of pseudo-labeled frames.
        unlabeled_list (list): List of unlabeled frames.
        info_dict (OrderedDict): Dictionary containing information about each frame.
        sequence_folders (list): List of sequence folders in the dataset.
    Methods:
        __init__(ratio, base_path, save_path, labeled_name, unlabeled_name, search_range):
            Initializes the ProgressivePLE class with the given parameters.
        load_poses(seq):
            Loads the poses for a given sequence.
        load_label(file):
            Loads the label data from a file.
        get_info_dict():
            Populates the info_dict with information about each frame.
        check_info_dict():
            Checks the consistency of the info_dict.
        run():
            Runs the PLE process on the dataset.
        do_ple(src_file, reference_list):
            Performs PLE on a source file using a list of reference files.
        do_ple_pseudo(src_file, reference_list_gt, reference_list_pseudo):
            Performs PLE on a source file using both ground truth and pseudo-labeled reference files.
    """

    def __init__(
        self,
        ratio: str = "0.5",
        base_path: str = "./",
        save_path: str = "./R0.5_PLE",
        labeled_name="semantickitti_infos_train.{}.pkl",
        unlabeled_name="semantickitti_infos_train.{}-unlabeled.pkl",
        search_range: int = 10,
    ):

        self.ratio = float(ratio)
        print(f"Progressive PLE with ratio {ratio}")
        self.base_path = base_path
        self.save_path = save_path
        self.labeled_name = labeled_name.format(ratio)
        self.unlabeled_name = unlabeled_name.format(ratio)
        self.search_range = search_range

        self.gt_labeled_list = []
        self.pseudo_labeled_list = []
        self.unlabeled_list = []

        self.info_dict = OrderedDict()
        self.sequence_folders = [
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "09",
            "10",
        ]

        self.get_info_dict()
        self.check_info_dict()

    def load_poses(self, seq):
        calib_path = os.path.join(self.base_path, "sequences", seq, "calib.txt")
        calibration = parse_calibration(calib_path)
        poses = parse_poses(
            os.path.join(self.base_path, "sequences", seq, "poses.txt"), calibration
        )
        return poses

    @staticmethod
    def load_label(file):
        label = np.fromfile(file, dtype=np.uint32).reshape((-1))
        return label

    def get_info_dict(self):
        for folder in self.sequence_folders:
            input_folder = os.path.join(self.base_path, "sequences", folder, "velodyne")
            scan_files = sorted(os.listdir(input_folder))
            self.info_dict[folder] = OrderedDict()
            for i, file in enumerate(scan_files):
                lidar_path = os.path.join("sequences", folder, "velodyne", file)
                self.info_dict[folder][lidar_path] = OrderedDict(
                    {
                        "lidar_path": lidar_path,
                        "label_type": None,
                        "seq": folder,
                        "#frame": i,
                    }
                )

        # TODO
        # 1. for each unlabeled lidar, find the nearest left and right labeled lidar points (less than 10 frames),
        # 2. if the nearest labeled lidar is less than 10 frames, then add the unlabeled lidar to reference_gt_list
        # 3. find the nearest pseudo labeled lidar points (less than 10 frames), then add the unlabeled lidar to reference_pseudo_list

        labeled_pkl = pickle.load(
            open(os.path.join(self.base_path, self.labeled_name), "rb")
        )
        unlabeled_pkl = pickle.load(
            open(os.path.join(self.base_path, self.unlabeled_name), "rb")
        )

        labeled_datalist = [
            d["lidar_points"]["lidar_path"] for d in labeled_pkl["data_list"]
        ]
        labeled_datalist.sort()
        unlabeled_datalist = [
            d["lidar_points"]["lidar_path"] for d in unlabeled_pkl["data_list"]
        ]
        unlabeled_datalist.sort()

        for unlabeled_lidar in unlabeled_datalist:
            # find the nearest left and right labeled lidar points (less than 10 frames)
            seq = get_seq(unlabeled_lidar)
            cur_frame = get_frame(unlabeled_lidar)
            # search right-side

            left_gt_lidar, right_gt_lidar = None, None
            for i in range(1, self.search_range + 1):
                nxt_frame = cur_frame + i
                if nxt_frame >= len(self.info_dict[seq]):
                    break
                nxt_lidar_path = os.path.join(
                    "sequences", seq, "velodyne", f"{nxt_frame:06d}.bin"
                )
                if nxt_lidar_path in labeled_datalist:
                    right_gt_lidar = i
                    break
            # search left-side
            for i in range(-1, -self.search_range - 1, -1):
                nxt_frame = cur_frame + i
                if nxt_frame < 0:
                    break
                nxt_lidar_path = os.path.join(
                    "sequences", seq, "velodyne", f"{nxt_frame:06d}.bin"
                )
                if nxt_lidar_path in labeled_datalist:
                    left_gt_lidar = i
                    break

            # get the nearest, if same distance, then choose both
            if left_gt_lidar is not None and right_gt_lidar is not None:
                if abs(left_gt_lidar) < right_gt_lidar:
                    self.info_dict[seq][unlabeled_lidar]["label_type"] = "pseudo"
                    self.info_dict[seq][unlabeled_lidar]["ref_gt"] = [
                        get_lidar_path(seq, cur_frame + left_gt_lidar)
                    ]
                    if (
                        get_lidar_path(seq, cur_frame + left_gt_lidar)
                        not in labeled_datalist
                    ):
                        raise ValueError(
                            f"{get_lidar_path(seq, cur_frame + left_gt_lidar)} not in labeled_datalist"
                        )
                    if left_gt_lidar != -1:
                        self.info_dict[seq][unlabeled_lidar]["ref_pseudo"] = [
                            get_lidar_path(seq, cur_frame + -1)
                        ]
                elif abs(left_gt_lidar) > right_gt_lidar:
                    self.info_dict[seq][unlabeled_lidar]["label_type"] = "pseudo"
                    self.info_dict[seq][unlabeled_lidar]["ref_gt"] = [
                        get_lidar_path(seq, cur_frame + right_gt_lidar)
                    ]
                    if (
                        get_lidar_path(seq, cur_frame + right_gt_lidar)
                        not in labeled_datalist
                    ):
                        raise ValueError(
                            f"{get_lidar_path(seq, cur_frame + right_gt_lidar)} not in labeled_datalist"
                        )
                    if right_gt_lidar != 1:
                        self.info_dict[seq][unlabeled_lidar]["ref_pseudo"] = [
                            get_lidar_path(seq, cur_frame + 1)
                        ]
                else:
                    self.info_dict[seq][unlabeled_lidar]["label_type"] = "pseudo"
                    self.info_dict[seq][unlabeled_lidar]["ref_gt"] = [
                        get_lidar_path(seq, cur_frame + left_gt_lidar),
                        get_lidar_path(seq, cur_frame + right_gt_lidar),
                    ]
                    if (
                        get_lidar_path(seq, cur_frame + left_gt_lidar)
                        not in labeled_datalist
                    ):
                        raise ValueError(
                            f"{get_lidar_path(seq, cur_frame + left_gt_lidar)} not in labeled_datalist"
                        )
                    if (
                        get_lidar_path(seq, cur_frame + right_gt_lidar)
                        not in labeled_datalist
                    ):
                        raise ValueError(
                            f"{get_lidar_path(seq, cur_frame + right_gt_lidar)} not in labeled_datalist"
                        )
                    if left_gt_lidar != -1:
                        self.info_dict[seq][unlabeled_lidar]["ref_pseudo"] = [
                            get_lidar_path(seq, cur_frame + -1)
                        ]
                    if right_gt_lidar != 1:
                        if self.info_dict[seq][unlabeled_lidar]["ref_pseudo"] is None:
                            self.info_dict[seq][unlabeled_lidar]["ref_pseudo"] = [
                                get_lidar_path(seq, cur_frame + 1)
                            ]
                        else:
                            self.info_dict[seq][unlabeled_lidar]["ref_pseudo"].append(
                                get_lidar_path(seq, cur_frame + 1)
                            )
            elif left_gt_lidar is not None and right_gt_lidar is None:
                self.info_dict[seq][unlabeled_lidar]["label_type"] = "pseudo"
                self.info_dict[seq][unlabeled_lidar]["ref_gt"] = [
                    get_lidar_path(seq, cur_frame + left_gt_lidar)
                ]
                if (
                    get_lidar_path(seq, cur_frame + left_gt_lidar)
                    not in labeled_datalist
                ):
                    raise ValueError(
                        f"{get_lidar_path(seq, cur_frame + left_gt_lidar)} not in labeled_datalist"
                    )

                if left_gt_lidar != -1:
                    self.info_dict[seq][unlabeled_lidar]["ref_pseudo"] = [
                        get_lidar_path(seq, cur_frame + -1)
                    ]
            elif right_gt_lidar is not None and left_gt_lidar is None:
                self.info_dict[seq][unlabeled_lidar]["label_type"] = "pseudo"
                self.info_dict[seq][unlabeled_lidar]["ref_gt"] = [
                    get_lidar_path(seq, cur_frame + right_gt_lidar)
                ]
                if (
                    get_lidar_path(seq, cur_frame + right_gt_lidar)
                    not in labeled_datalist
                ):
                    raise ValueError(
                        f"{get_lidar_path(seq, cur_frame + right_gt_lidar)} not in labeled_datalist"
                    )

                if right_gt_lidar != 1:
                    self.info_dict[seq][unlabeled_lidar]["ref_pseudo"] = [
                        get_lidar_path(seq, cur_frame + 1)
                    ]
            else:
                self.info_dict[seq][unlabeled_lidar]["label_type"] = "unlabeled"

    def check_info_dict(self):
        labeled_pkl = pickle.load(
            open(os.path.join(self.base_path, self.labeled_name), "rb")
        )
        unlabeled_pkl = pickle.load(
            open(os.path.join(self.base_path, self.unlabeled_name), "rb")
        )

        labeled_datalist = [
            d["lidar_points"]["lidar_path"] for d in labeled_pkl["data_list"]
        ]
        labeled_datalist.sort()
        unlabeled_datalist = [
            d["lidar_points"]["lidar_path"] for d in unlabeled_pkl["data_list"]
        ]
        unlabeled_datalist.sort()

        for seq in self.info_dict:
            for lidar_path in self.info_dict[seq]:
                if self.info_dict[seq][lidar_path]["label_type"] == "pseudo":
                    for ref_gt in self.info_dict[seq][lidar_path]["ref_gt"]:
                        if ref_gt not in labeled_datalist:
                            raise ValueError(f"{ref_gt} has no ref_gt")
                    if "ref_pseudo" in self.info_dict[seq][lidar_path]:
                        for ref_pseudo in self.info_dict[seq][lidar_path]["ref_pseudo"]:
                            if ref_pseudo not in unlabeled_datalist:
                                raise ValueError(f"{ref_pseudo} has no ref_pseudo")

    def run(self):
        # get pseudo_file_list from info_dict which has no 'ref_pseudo'
        pseudo_file_list = []
        for seq in self.info_dict:
            for lidar_path in self.info_dict[seq]:
                if (
                    self.info_dict[seq][lidar_path]["label_type"] == "pseudo"
                    and "ref_pseudo" not in self.info_dict[seq][lidar_path]
                ):
                    pseudo_file_list.append(lidar_path)

        made_list = []
        print("Do PLE on files from Ref GT")
        # do ple
        for pseudo_lidar_path in tqdm(pseudo_file_list):
            self.do_ple(
                pseudo_lidar_path,
                self.info_dict[get_seq(pseudo_lidar_path)][pseudo_lidar_path]["ref_gt"],
            )
            made_list.append(pseudo_lidar_path)

        # print('Do PLE on files from Ref Pseudo')
        pseudo_ref_list = deque()
        for seq in self.info_dict:
            for lidar_path in self.info_dict[seq]:
                if (
                    self.info_dict[seq][lidar_path]["label_type"] == "pseudo"
                    and "ref_pseudo" in self.info_dict[seq][lidar_path]
                ):
                    pseudo_ref_list.append(lidar_path)

        pseudo_ref_file_ordered = list()
        while len(pseudo_ref_list) > 0:

            pseudo_lidar_path = pseudo_ref_list.popleft()
            seq = get_seq(pseudo_lidar_path)
            reference_list_gt = self.info_dict[get_seq(pseudo_lidar_path)][
                pseudo_lidar_path
            ]["ref_gt"]
            reference_list_pseudo = self.info_dict[get_seq(pseudo_lidar_path)][
                pseudo_lidar_path
            ]["ref_pseudo"]

            all_exist = True
            for ref_pseudo in reference_list_pseudo:
                # ref_pseudo_path = f'{self.save_path}/{seq}/labels/{get_frame(ref_pseudo):06d}.label'
                if ref_pseudo not in made_list:
                    all_exist = False
                    break
                # if not os.path.exists(ref_pseudo_path):
                #     all_exist = False
                #     break
            if all_exist:
                # self.do_ple_pseudo(pseudo_lidar_path, reference_list_gt, reference_list_pseudo)
                made_list.append(pseudo_lidar_path)
                pseudo_ref_file_ordered.append(pseudo_lidar_path)
            else:
                pseudo_ref_list.append(pseudo_lidar_path)

        for pseudo_lidar_path in tqdm(pseudo_ref_file_ordered):
            reference_list_gt = self.info_dict[get_seq(pseudo_lidar_path)][
                pseudo_lidar_path
            ]["ref_gt"]
            reference_list_pseudo = self.info_dict[get_seq(pseudo_lidar_path)][
                pseudo_lidar_path
            ]["ref_pseudo"]
            self.do_ple_pseudo(
                pseudo_lidar_path, reference_list_gt, reference_list_pseudo
            )

    def do_ple(self, src_file, reference_list):

        seq = get_seq(src_file)
        poses = self.load_poses(seq)

        src_path = os.path.join(self.base_path, src_file)
        src_scan = np.fromfile(src_path, dtype=np.float32).reshape(-1, 4)
        src_pose = poses[get_frame(src_path)]

        out_points = []
        out_labels = []
        for ref_file in reference_list:
            ref_file = os.path.join(self.base_path, ref_file)
            ref_scan = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 4)
            ref_label = ref_file.replace("velodyne", "labels").replace(".bin", ".label")
            ref_label = self.load_label(ref_label)
            ref_pose = poses[get_frame(ref_file)]

            ref_points = np.ones(ref_scan.shape)
            ref_points[:, :3] = ref_scan[:, :3]
            diff = np.matmul(inv(src_pose), ref_pose)
            transformed_points = np.matmul(diff, ref_points.transpose()).transpose()
            out_points.append(transformed_points[:, :3])
            out_labels.append(ref_label)
        out_points = np.concatenate(out_points, axis=0)
        out_labels = np.concatenate(out_labels, axis=0)

        coord1 = src_scan[:, :3]
        coord2 = out_points

        kd_tree = KDTree(coord2)
        _, idx = kd_tree.query(coord1, k=1, workers=16)

        src_label = out_labels[idx]
        save_file_path = os.path.join(
            self.save_path,
            src_file.replace("velodyne", "labels").replace(".bin", ".label"),
        )
        save_file_path = save_file_path.replace("/sequences", "")
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        src_label.tofile(save_file_path)

    def do_ple_pseudo(self, src_file, reference_list_gt, reference_list_pseudo):

        seq = get_seq(src_file)
        poses = self.load_poses(seq)

        src_path = os.path.join(self.base_path, src_file)
        src_scan = np.fromfile(src_path, dtype=np.float32).reshape(-1, 4)
        src_pose = poses[get_frame(src_path)]

        out_points = []
        out_labels = []
        for ref_file in reference_list_gt:
            ref_file = os.path.join(self.base_path, ref_file)
            ref_scan = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 4)
            ref_label = ref_file.replace("velodyne", "labels").replace(".bin", ".label")
            ref_label = self.load_label(ref_label)
            ref_pose = poses[get_frame(ref_file)]

            ref_points = np.ones(ref_scan.shape)
            ref_points[:, :3] = ref_scan[:, :3]
            diff = np.matmul(inv(src_pose), ref_pose)
            transformed_points = np.matmul(diff, ref_points.transpose()).transpose()
            out_points.append(transformed_points[:, :3])
            out_labels.append(ref_label)
        for ref_file in reference_list_pseudo:
            ref_file = os.path.join(self.base_path, ref_file)
            ref_scan = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 4)
            ref_label = f"{self.save_path}/{seq}/labels/{get_frame(ref_file):06d}.label"
            ref_label = self.load_label(ref_label)
            ref_pose = poses[get_frame(ref_file)]

            ref_points = np.ones(ref_scan.shape)
            ref_points[:, :3] = ref_scan[:, :3]
            diff = np.matmul(inv(src_pose), ref_pose)
            transformed_points = np.matmul(diff, ref_points.transpose()).transpose()
            out_points.append(transformed_points[:, :3])
            out_labels.append(ref_label)
        out_points = np.concatenate(out_points, axis=0)
        out_labels = np.concatenate(out_labels, axis=0)

        coord1 = src_scan[:, :3]
        coord2 = out_points

        kd_tree = KDTree(coord2)
        _, idx = kd_tree.query(coord1, k=1, workers=16)

        src_label = out_labels[idx]
        save_file_path = os.path.join(
            self.save_path,
            src_file.replace("velodyne", "labels").replace(".bin", ".label"),
        )
        save_file_path = save_file_path.replace("/sequences", "")
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        src_label.tofile(save_file_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ratio", type=str, default=0.5, help="Ratio of labeled data")
    parser.add_argument(
        "--base_path",
        type=str,
        default="~/dataset/SemanticKITTI/dataset/",
        help="Path to the SemanticKITTI dataset",
    )
    parser.add_argument("--save_path", type=str, default="R0.5_PLE")
    parser.add_argument(
        "--labeled_name",
        type=str,
        default="semantickitti_infos_train.{}.pkl",
        help="Template name for labeled data files.",
    )
    parser.add_argument(
        "--unlabeled_name",
        type=str,
        default="semantickitti_infos_train.{}-unlabeled.pkl",
        help="Template name for unlabeled data files.",
    )
    parser.add_argument(
        "--search_range",
        type=int,
        default=10,
        help="Range to search for nearest labeled",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ple_generator = ProgressivePLE(
        ratio=args.ratio,
        base_path=args.base_path,
        save_path=args.save_path,
        labeled_name=args.labeled_name,
        unlabeled_name=args.unlabeled_name,
    )
    ple_generator.run()
