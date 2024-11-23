import argparse
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm


class IoUEvaluator:
    def __init__(self, n_classes, ignore=None):
        # classes
        self.n_classes = n_classes

        # What to include and ignore from the means
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64
        )
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)
        self.conf_matrix = None
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def add_batch(self, x_pred, x_gt):
        x_row = x_pred.reshape(-1)
        y_row = x_gt.reshape(-1)
        assert x_row.shape == y_row.shape

        # create indexes
        idxs = tuple(np.stack((x_row, y_row), axis=0))
        np.add.at(self.conf_matrix, idxs, 1)

    def get_stats(self):
        # remove fp from confusion on the ignore classes cols
        conf = self.conf_matrix.copy()
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def get_iou(self):
        tp, fp, fn = self.get_stats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def get_acc(self):
        tp, fp, fn = self.get_stats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"

    def get_confusion(self):
        return self.conf_matrix.copy()


def parse_argument():
    parser = argparse.ArgumentParser("./nuscenes_03_evaluate.py")
    parser.add_argument("--gt", type=str, default="", help="Path to the ground truth")
    parser.add_argument("--pred", type=str, default="", help="Path to the predictions")
    parser.add_argument("--output", type=str, default=None, help="Path to the output")
    parser.add_argument(
        "--config",
        type=str,
        default="semantic-nuscenes.yaml",
        help="Path to the config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    if args.output is None:
        args.output = args.pred + "_result.txt"
    CONFIG = yaml.safe_load(open(args.config, "r"))

    with open(args.output, "w") as f:
        sys.stdout = f
        print("nuScenes evaluation")
        print("Predictions: ", args.pred)
        print("Ground Truth: ", args.gt)

        # get information from config file
        class_strings = CONFIG["labels"]
        class_remap = CONFIG["learning_map"]
        class_inv_remap = CONFIG["learning_map_inv"]
        class_ignore = CONFIG["learning_ignore"]
        num_classes = len(class_inv_remap)
        evaluation_sequences = CONFIG["split"]["train"]

        # make lookup table for mapping
        maxkey = max(class_remap.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(class_remap.keys())] = list(class_remap.values())

        # create evaluator
        ignore = []
        for cl, ign in class_ignore.items():
            if ign:
                x_cl = int(cl)
                ignore.append(x_cl)
                print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

        evaluator = IoUEvaluator(num_classes, ignore)
        evaluator.reset()

        # get predictions paths
        pred_names = []
        for sequence in evaluation_sequences:
            sequence = "{0:04d}".format(int(sequence))
            pred_paths = os.path.join(args.pred, sequence, "labels")
            # populate the label names
            seq_pred_names = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(pred_paths))
                for f in fn
                if ".label" in f
            ]
            seq_pred_names.sort()
            pred_names.extend(seq_pred_names)
        print("Found ", len(pred_names), " predictions")
        # get label paths (partial)
        folder_name = args.pred.split("/")[-1]
        label_names = []
        for pred in pred_names:
            gt = os.path.join(
                args.gt, "sequences", pred.split("/")[-3], "labels", pred.split("/")[-1]
            )
            # label = pred.replace(folder_name, 'sequences')
            label_names.append(gt)

        # check that I have the same number of files
        print("Checking number of files...")
        print("labels: ", len(label_names))
        print("predictions: ", len(pred_names))
        assert len(label_names) == len(pred_names)

        count = 0

        # open each file, get the tensor, and make the iou comparison
        for label_file, pred_file in tqdm(zip(label_names, pred_names)):
            count += 1
            # open label
            label = np.fromfile(label_file, dtype=np.uint8)
            label = label.reshape((-1))  # reshape to vector
            # label = label & 0xFFFF  # get lower half for semantics
            label = remap_lut[label]  # remap to xentropy format

            # open prediction
            pred = np.fromfile(pred_file, dtype=np.uint8)
            pred = pred.reshape((-1))  # reshape to vector
            # pred = pred & 0xFFFF  # get lower half for semantics
            pred = remap_lut[pred]  # remap to xentropy format

            evaluator.add_batch(pred, label)

        # Calculate Precision
        epsilon = np.finfo(np.float32).eps
        conf = evaluator.get_confusion()
        # calculate precision
        tp, fp, fn = evaluator.get_stats()

        effective_conf = conf[1:, 1:]
        recall = np.diag(effective_conf) / (np.sum(effective_conf, axis=1) + epsilon)
        precision = np.diag(effective_conf) / (np.sum(effective_conf, axis=0) + epsilon)
        iou = np.diag(effective_conf) / (
            np.sum(effective_conf, axis=1)
            + np.sum(effective_conf, axis=0)
            - np.diag(effective_conf)
            + epsilon
        )
        acc = np.diag(effective_conf).sum() / effective_conf.sum()

        # pretty print
        print(
            f'{"Class Name":<20}{"Precision (%)":>15}{"Recall (%)":>15}{"IoU (%)":>15}{"Accuracy (%)":>15}'
        )

        # average performance
        print(
            f'{"Mean":<20}{np.mean(precision) * 100:>15.2f}{np.mean(recall) * 100:>15.2f}{np.mean(iou) * 100:>15.2f}{acc * 100:>15.2f}'
        )

        # each class
        for i in range(1, num_classes):
            class_name = class_strings[class_inv_remap[i]]
            prec = precision[i - 1] * 100
            rec = recall[i - 1] * 100
            iou_ = iou[i - 1] * 100

            print(f"{class_name:<20}{prec:>15.2f}{rec:>15.2f}{iou_:>15.2f}")

        sys.stdout = sys.__stdout__
