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
    """ read calibration file with given filename

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
    """ read poses file with per-scan poses from given filename

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
    return lidar_path.split('/')[1]


def get_frame(lidar_path):
    return int(lidar_path.split('/')[-1].split('.')[0])

def get_lidar_path(seq, frame):
    return os.path.join('sequences', seq, 'velodyne', f'{frame:06d}.bin')


class ProgressivePLE:
    def __init__(self,
                 ratio=0.5,
                 base_path='~/dataset/nuscenes_kitti',
                 save_path='EXE',
                 labeled_name='nuscenes_kitti_infos_train.{}.pkl',
                 unlabeled_name='nuscenes_kitti_infos_train.{}-unlabeled.pkl',
                 search_range=2,
                 ):

        self.ratio = float(ratio)
        print(f'Progressive PLE with ratio {ratio}')
        self.base_path = base_path
        self.save_path = save_path
        self.labeled_name = labeled_name.format(ratio)
        self.unlabeled_name = unlabeled_name.format(ratio)
        self.search_range = search_range
        self.save_path = save_path

        self.gt_labeled_list = []
        self.pseudo_labeled_list = []
        self.unlabeled_list = []

        self.info_dict = OrderedDict()
        # self.sequence_folders = sorted(os.listdir(os.path.join(self.base_path, 'sequences')))
        self.sequence_folders = sorted(f'{i:04d}' for i in [1,2,4,5,6,7,8,9,10,11,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,138,139,149,150,151,152,154,155,157,158,159,160,161,162,163,164,165,166,167,168,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,187,188,190,191,192,193,194,195,196,199,200,202,203,204,206,207,208,209,210,211,212,213,214,218,219,220,222,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,315,316,317,318,321,323,324,328,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,405,406,407,408,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,461,462,463,464,465,467,468,469,471,472,474,475,476,477,478,479,480,499,500,501,502,504,505,506,507,508,509,510,511,512,513,514,515,517,518,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,541,542,543,544,545,546,566,568,570,571,572,573,574,575,576,577,578,580,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,681,683,684,685,686,687,688,689,695,696,697,698,700,701,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,726,727,728,730,731,733,734,735,736,737,738,739,740,741,744,746,747,749,750,751,752,757,758,759,760,761,762,763,764,765,767,768,769,786,787,789,790,791,792,803,804,805,806,808,809,810,811,812,813,815,816,817,819,820,821,822,847,848,849,850,851,852,853,854,855,856,858,860,861,862,863,864,865,866,868,869,870,871,872,873,875,876,877,878,880,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,945,947,949,952,953,955,956,957,958,959,960,961,975,976,977,978,979,980,981,982,983,984,988,989,990,991,992,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1104,1105,1106,1107,1108,1109,1110])

        self.get_info_dict()
        self.check_info_dict()

    def load_poses(self, seq):
        calib_path = os.path.join(self.base_path, 'sequences', seq, 'calib.txt')
        calibration = parse_calibration(calib_path)
        poses = parse_poses(os.path.join(self.base_path, 'sequences', seq, "poses.txt"), calibration)
        return poses

    @staticmethod
    def load_label(file):
        label = np.fromfile(file, dtype=np.uint8).reshape((-1))
        return label

    def get_info_dict(self):
        print('\nGetting final labeled datalist...')
        labeled_pkl = pickle.load(open(os.path.join(self.base_path, self.labeled_name), "rb"))
        unlabeled_pkl = pickle.load(open(os.path.join(self.base_path, self.unlabeled_name), "rb"))
        labeled_datalist = [d['lidar_points']['lidar_path'] for d in labeled_pkl['data_list']]
        labeled_datalist.sort()
        unlabeled_datalist = [d['lidar_points']['lidar_path'] for d in unlabeled_pkl['data_list']]
        unlabeled_datalist.sort()

        train_datalist = labeled_datalist + unlabeled_datalist
        for folder in self.sequence_folders:
            input_folder = os.path.join(self.base_path, 'sequences', folder, "velodyne")
            scan_files = sorted(os.listdir(input_folder))
            self.info_dict[folder] = OrderedDict()
            for i, file in enumerate(scan_files):
                lidar_path = os.path.join('sequences', folder, 'velodyne', file)
                if lidar_path not in train_datalist:
                    continue
                self.info_dict[folder][lidar_path] = OrderedDict({'lidar_path': lidar_path,
                                                                  'label_type': None,
                                                                  'seq': folder,
                                                                  '#frame': i})

        # TODO
        # 1. for each unlabeled lidar, find the nearest left and right labeled lidar points (less than 10 frames),
        # 2. if the nearest labeled lidar is less than 10 frames, then add the unlabeled lidar to reference_gt_list
        # 3. find the nearest pseudo labeled lidar points (less than 10 frames), then add the unlabeled lidar to reference_pseudo_list

        # labeled_pkl = pickle.load(open(os.path.join(self.base_path, self.labeled_name), "rb"))
        # unlabeled_pkl = pickle.load(open(os.path.join(self.base_path, self.unlabeled_name), "rb"))
        #
        # labeled_datalist = [d['lidar_points']['lidar_path'] for d in labeled_pkl['data_list']]
        # labeled_datalist.sort()
        # unlabeled_datalist = [d['lidar_points']['lidar_path'] for d in unlabeled_pkl['data_list']]
        # unlabeled_datalist.sort()

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
                nxt_lidar_path = os.path.join('sequences', seq, 'velodyne', f'{nxt_frame:06d}.bin')
                if nxt_lidar_path in labeled_datalist:
                    right_gt_lidar = i
                    break
            # search left-side
            for i in range(-1, -self.search_range - 1, -1):
                nxt_frame = cur_frame + i
                if nxt_frame < 0:
                    break
                nxt_lidar_path = os.path.join('sequences', seq, 'velodyne', f'{nxt_frame:06d}.bin')
                if nxt_lidar_path in labeled_datalist:
                    left_gt_lidar = i
                    break

            # get the nearest, if same distance, then choose both
            if left_gt_lidar is not None and right_gt_lidar is not None:
                if abs(left_gt_lidar) < right_gt_lidar:
                    self.info_dict[seq][unlabeled_lidar]['label_type'] = 'pseudo'
                    self.info_dict[seq][unlabeled_lidar]['ref_gt'] = [get_lidar_path(seq, cur_frame + left_gt_lidar)]
                    if get_lidar_path(seq, cur_frame + left_gt_lidar) not in labeled_datalist:
                        raise ValueError(f'{get_lidar_path(seq, cur_frame + left_gt_lidar)} not in labeled_datalist')
                    if left_gt_lidar != -1:
                        self.info_dict[seq][unlabeled_lidar]['ref_pseudo'] = [get_lidar_path(seq, cur_frame + -1)]
                elif abs(left_gt_lidar) > right_gt_lidar:
                    self.info_dict[seq][unlabeled_lidar]['label_type'] = 'pseudo'
                    self.info_dict[seq][unlabeled_lidar]['ref_gt'] = [get_lidar_path(seq, cur_frame + right_gt_lidar)]
                    if get_lidar_path(seq, cur_frame + right_gt_lidar) not in labeled_datalist:
                        raise ValueError(f'{get_lidar_path(seq, cur_frame + right_gt_lidar)} not in labeled_datalist')
                    if right_gt_lidar != 1:
                        self.info_dict[seq][unlabeled_lidar]['ref_pseudo'] = [get_lidar_path(seq, cur_frame + 1)]
                else:
                    self.info_dict[seq][unlabeled_lidar]['label_type'] = 'pseudo'
                    self.info_dict[seq][unlabeled_lidar]['ref_gt'] = [get_lidar_path(seq, cur_frame + left_gt_lidar),
                                                                      get_lidar_path(seq, cur_frame + right_gt_lidar)]
                    if get_lidar_path(seq, cur_frame + left_gt_lidar) not in labeled_datalist:
                        raise ValueError(f'{get_lidar_path(seq, cur_frame + left_gt_lidar)} not in labeled_datalist')
                    if get_lidar_path(seq, cur_frame + right_gt_lidar) not in labeled_datalist:
                        raise ValueError(f'{get_lidar_path(seq, cur_frame + right_gt_lidar)} not in labeled_datalist')
                    if left_gt_lidar != -1:
                        self.info_dict[seq][unlabeled_lidar]['ref_pseudo'] = [get_lidar_path(seq, cur_frame + -1)]
                    if right_gt_lidar != 1:
                        if self.info_dict[seq][unlabeled_lidar]['ref_pseudo'] is None:
                            self.info_dict[seq][unlabeled_lidar]['ref_pseudo'] = [get_lidar_path(seq, cur_frame + 1)]
                        else:
                            self.info_dict[seq][unlabeled_lidar]['ref_pseudo'].append(get_lidar_path(seq, cur_frame + 1))
            elif left_gt_lidar is not None and right_gt_lidar is None:
                self.info_dict[seq][unlabeled_lidar]['label_type'] = 'pseudo'
                self.info_dict[seq][unlabeled_lidar]['ref_gt'] = [get_lidar_path(seq, cur_frame + left_gt_lidar)]
                if get_lidar_path(seq, cur_frame + left_gt_lidar) not in labeled_datalist:
                    raise ValueError(f'{get_lidar_path(seq, cur_frame + left_gt_lidar)} not in labeled_datalist')

                if left_gt_lidar != -1:
                    self.info_dict[seq][unlabeled_lidar]['ref_pseudo'] = [get_lidar_path(seq, cur_frame + -1)]
            elif right_gt_lidar is not None and left_gt_lidar is None:
                self.info_dict[seq][unlabeled_lidar]['label_type'] = 'pseudo'
                self.info_dict[seq][unlabeled_lidar]['ref_gt'] = [get_lidar_path(seq, cur_frame + right_gt_lidar)]
                if get_lidar_path(seq, cur_frame + right_gt_lidar) not in labeled_datalist:
                    raise ValueError(f'{get_lidar_path(seq, cur_frame + right_gt_lidar)} not in labeled_datalist')

                if right_gt_lidar != 1:
                    self.info_dict[seq][unlabeled_lidar]['ref_pseudo'] = [get_lidar_path(seq, cur_frame + 1)]
            else:
                self.info_dict[seq][unlabeled_lidar]['label_type'] = 'unlabeled'

    def check_info_dict(self):
        labeled_pkl = pickle.load(open(os.path.join(self.base_path, self.labeled_name), "rb"))
        unlabeled_pkl = pickle.load(open(os.path.join(self.base_path, self.unlabeled_name), "rb"))

        labeled_datalist = [d['lidar_points']['lidar_path'] for d in labeled_pkl['data_list']]
        labeled_datalist.sort()
        unlabeled_datalist = [d['lidar_points']['lidar_path'] for d in unlabeled_pkl['data_list']]
        unlabeled_datalist.sort()

        for seq in self.info_dict:
            for lidar_path in self.info_dict[seq]:
                if self.info_dict[seq][lidar_path]['label_type'] == 'pseudo':
                    for ref_gt in self.info_dict[seq][lidar_path]['ref_gt']:
                        if ref_gt not in labeled_datalist:
                            raise ValueError(f'{ref_gt} has no ref_gt')
                    if 'ref_pseudo' in self.info_dict[seq][lidar_path]:
                        for ref_pseudo in self.info_dict[seq][lidar_path]['ref_pseudo']:
                            if ref_pseudo not in unlabeled_datalist:
                                raise ValueError(f'{ref_pseudo} has no ref_pseudo')

    def run(self):
        # get pseudo_file_list from info_dict which has no 'ref_pseudo'
        pseudo_file_list = []
        for seq in self.info_dict:
            for lidar_path in self.info_dict[seq]:
                if self.info_dict[seq][lidar_path]['label_type'] == 'pseudo' and 'ref_pseudo' not in self.info_dict[seq][lidar_path]:
                    pseudo_file_list.append(lidar_path)

        made_list = []
        print('Do PLE on files from Ref GT')
        # do ple
        for pseudo_lidar_path in tqdm(pseudo_file_list):
            self.do_ple(pseudo_lidar_path, self.info_dict[get_seq(pseudo_lidar_path)][pseudo_lidar_path]['ref_gt'])
            made_list.append(pseudo_lidar_path)

        # print('Do PLE on files from Ref Pseudo')
        pseudo_ref_list = deque()
        for seq in self.info_dict:
            for lidar_path in self.info_dict[seq]:
                if self.info_dict[seq][lidar_path]['label_type'] == 'pseudo' and 'ref_pseudo' in self.info_dict[seq][lidar_path]:
                    pseudo_ref_list.append(lidar_path)

        pseudo_ref_file_ordered = list()
        while len(pseudo_ref_list) > 0:

            pseudo_lidar_path = pseudo_ref_list.popleft()
            seq = get_seq(pseudo_lidar_path)
            reference_list_gt = self.info_dict[get_seq(pseudo_lidar_path)][pseudo_lidar_path]['ref_gt']
            reference_list_pseudo = self.info_dict[get_seq(pseudo_lidar_path)][pseudo_lidar_path]['ref_pseudo']

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
            reference_list_gt = self.info_dict[get_seq(pseudo_lidar_path)][pseudo_lidar_path]['ref_gt']
            reference_list_pseudo = self.info_dict[get_seq(pseudo_lidar_path)][pseudo_lidar_path]['ref_pseudo']
            self.do_ple_pseudo(pseudo_lidar_path, reference_list_gt, reference_list_pseudo)

    def do_ple(self, src_file, reference_list):

        seq = get_seq(src_file)
        poses = self.load_poses(seq)

        src_path = os.path.join(self.base_path, src_file)
        src_scan = np.fromfile(src_path, dtype=np.float32).reshape(-1, 5)
        src_pose = poses[get_frame(src_path)]

        out_points = []
        out_labels = []
        for ref_file in reference_list:
            ref_file = os.path.join(self.base_path, ref_file)
            ref_scan = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 5)
            ref_label = ref_file.replace('velodyne', 'labels').replace('.bin', '.label')
            ref_label = self.load_label(ref_label)
            ref_pose = poses[get_frame(ref_file)]

            ref_points = np.ones((ref_scan.shape[0], 4))
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
        save_file_path = os.path.join(self.save_path, src_file.replace('velodyne', 'labels').replace('.bin', '.label'))
        save_file_path = save_file_path.replace('/sequences', '')
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        src_label.tofile(save_file_path)

    def do_ple_pseudo(self, src_file, reference_list_gt, reference_list_pseudo):

        seq = get_seq(src_file)
        poses = self.load_poses(seq)

        src_path = os.path.join(self.base_path, src_file)
        src_scan = np.fromfile(src_path, dtype=np.float32).reshape(-1, 5)
        src_pose = poses[get_frame(src_path)]

        out_points = []
        out_labels = []
        for ref_file in reference_list_gt:
            ref_file = os.path.join(self.base_path, ref_file)
            ref_scan = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 5)
            ref_label = ref_file.replace('velodyne', 'labels').replace('.bin', '.label')
            ref_label = self.load_label(ref_label)
            ref_pose = poses[get_frame(ref_file)]

            ref_points = np.ones((ref_scan.shape[0], 4))
            ref_points[:, :3] = ref_scan[:, :3]
            diff = np.matmul(inv(src_pose), ref_pose)
            transformed_points = np.matmul(diff, ref_points.transpose()).transpose()
            out_points.append(transformed_points[:, :3])
            out_labels.append(ref_label)
        for ref_file in reference_list_pseudo:
            ref_file = os.path.join(self.base_path, ref_file)
            ref_scan = np.fromfile(ref_file, dtype=np.float32).reshape(-1, 5)
            ref_label = f'{self.save_path}/{seq}/labels/{get_frame(ref_file):06d}.label'
            ref_label = self.load_label(ref_label)
            ref_pose = poses[get_frame(ref_file)]

            ref_points = np.ones((ref_scan.shape[0], 4))
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
        src_label = src_label.astype(np.uint8)
        save_file_path = os.path.join(self.save_path, src_file.replace('velodyne', 'labels').replace('.bin', '.label'))
        save_file_path = save_file_path.replace('/sequences', '')
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        src_label.tofile(save_file_path)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ratio', type=str,
                        default='0.5', help='labeled ratio')
    parser.add_argument('--base_path', type=str,
                        default='', help='base path')
    parser.add_argument('--save_path', type=str,
                        default='', help='save path')
    parser.add_argument('--labeled_name', type=str,
                        default='nuscenes_kitti_infos_train.{}.pkl', help='labeled_name')
    parser.add_argument('--unlabeled_name', type=str,
                        default='nuscenes_kitti_infos_train.{}-unlabeled.pkl')
    parser.add_argument('--search_range', type=int, default=2, help='search range')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ple_generator = ProgressivePLE(
        ratio=args.ratio,
        base_path=args.base_path,
        save_path=args.save_path,
        labeled_name=args.labeled_name,
        unlabeled_name=args.unlabeled_name,
        search_range=args.search_range
    )
    ple_generator.run()
