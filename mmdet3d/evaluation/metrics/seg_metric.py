# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import logging

import tempfile
from typing import Dict, Optional, Sequence
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.dist import (broadcast_object_list, is_main_process)
from mmengine.structures import BaseDataElement

from mmdet3d.evaluation import seg_eval
from mmdet3d.registry import METRICS


# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
from collections import OrderedDict
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)
from torch.distributed import ProcessGroup

import mmengine
from mmengine.dist.utils import (get_world_size, get_rank, get_backend, get_dist_info,
                            get_default_group, barrier, get_data_device,
                    get_comm_device, cast_data_device)
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.device import is_npu_available


def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data


@METRICS.register_module()
class SegMetric(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 collect_dir: str = None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.collect_dir = collect_dir
        super(SegMetric, self).__init__(
            prefix=prefix, collect_device=collect_device, collect_dir=collect_dir)

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_pts_seg']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        if submission_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            submission_prefix = osp.join(tmp_dir.name, 'results')
        mmengine.mkdir_or_exist(submission_prefix)

        # print(len(results))
        # print(results[0][0].keys())
        # print(results[0][0]['pts_semantic_mask'].shape, results[0][0]['pts_semantic_mask'].max())
        # print(results[0][1]['pts_semantic_mask'].shape, results[0][1]['pts_semantic_mask'].max())
        # print(self.dataset_meta.keys())
        seg_label_mapping = self.dataset_meta['seg_label_mapping']
        # print(seg_label_mapping)
        # print(self.dataset_meta['DATASET'])
        dataset_name = self.dataset_meta['DATASET']
        ignore_index = self.dataset_meta['ignore_index']
        # seg_label_mapping contains label2cat
        cat2label = np.zeros(len(seg_label_mapping)).astype(np.int32)
        for original_label, output_idx in seg_label_mapping.items():
            if output_idx != ignore_index:
                cat2label[output_idx] = original_label

        for i, (eval_ann, result) in enumerate(results):
            mask_name = eval_ann['mask_name']
            pred_sem_mask = result['pts_semantic_mask']
            if dataset_name == 'SemanticKITTI':
                pred_sem_mask = pred_sem_mask.astype(np.int32)
            else:
                pred_sem_mask = pred_sem_mask.astype(np.uint8)
            pred_label = cat2label[pred_sem_mask]
            curr_file = f'{submission_prefix}/{mask_name}'
            folder = osp.dirname(curr_file)
            mmengine.mkdir_or_exist(folder)
            pred_label.tofile(curr_file)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            # return None

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']

        gt_semantic_masks = []
        pred_semantic_masks = []

        for eval_ann, sinlge_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            pred_semantic_masks.append(
                sinlge_pred_results['pts_semantic_mask'])

        ret_dict = seg_eval(
            gt_semantic_masks,
            pred_semantic_masks,
            label2cat,
            ignore_index,
            logger=logger)

        return ret_dict


def collect_results(results: list,
                    size: int,
                    device: str = 'cpu',
                    tmpdir: Optional[str] = None) -> Optional[list]:
    """Collected results in distributed environments.

    Args:
        results (list[object]): Result list containing result parts to be
            collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        device (str): Device name. Optional values are 'cpu' and 'gpu'.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a temporal directory for it.
            ``tmpdir`` should be None when device is 'gpu'. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results(data, size, device='cpu')
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    if device not in ['gpu', 'cpu']:
        raise NotImplementedError(
            f"device must be 'cpu' or 'gpu', but got {device}")

    if device == 'gpu':
        assert tmpdir is None, 'tmpdir should be None when device is "gpu"'
        return collect_results_gpu(results, size)
    else:
        return collect_results_cpu(results, size, tmpdir)


def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_cpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
        if rank == 0:
            mmengine.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:  # type: ignore
        pickle.dump(result_part, f, protocol=2)

    barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            if not osp.exists(path):
                raise FileNotFoundError(
                    f'{tmpdir} is not an shared directory for '
                    f'rank {i}, please make sure {tmpdir} is a shared '
                    'directory for all ranks!')
            with open(path, 'rb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list[object]): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_gpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # gather all result part. Note that NCCL does not support gather so use
    # all_gather_object instead.
    part_list = all_gather_object(result_part)

    if rank == 0:
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None

def broadcast(data: Tensor,
              src: int = 0,
              group: Optional[ProcessGroup] = None) -> None:
    """Broadcast the data from ``src`` process to the whole group.

    ``data`` must have the same number of elements in all processes
    participating in the collective.

    Note:
        Calling ``broadcast`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Data to be sent if ``src`` is the rank of current
            process, and data to be used to save received data otherwise.
        src (int): Source rank. Defaults to 0.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> dist.broadcast(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.broadcast(data)
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([1, 2]) # Rank 1
    """
    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)
        # broadcast requires tensor is contiguous
        data_on_device = data_on_device.contiguous()  # type: ignore
        torch_dist.broadcast(data_on_device, src, group)

        if get_rank(group) != src:
            cast_data_device(data_on_device, input_device, data)



def _all_gather_object(object_list: List[Any],
                       obj: Any,
                       group: Optional[ProcessGroup] = None) -> None:
    """Gather picklable objects from the whole group into a list.

    Similar to :func:`all_gather`, but Python objects can be passed in.
    Note that the object must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as
            the size of the group for this collective and will contain the
            output.
        object (Any): Pickable Python object to be broadcast from current
            process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list``
        will be unmodified.
    """
    if torch_dist.distributed_c10d._rank_not_in_group(group):
        return

    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = get_backend(group)
    current_device = torch.device('cpu')
    is_nccl_backend = group_backend == torch_dist.Backend.NCCL
    if is_nccl_backend:
        # See note about using torch.cuda.current_device() here in docstring.
        # We cannot simply use my_rank since rank == device is not necessarily
        # true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        input_tensor = input_tensor.to(current_device)
        local_size = local_size.to(current_device)
    # Gather all local sizes. This is so that we can find the max size, and
    # index until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device)
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    torch_dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device)
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    torch_dist.all_gather(output_tensors, input_tensor, group=group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


def all_gather_object(data: Any,
                      group: Optional[ProcessGroup] = None) -> List[Any]:
    """Gather picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Note:
        Calling ``all_gather_object`` in non-distributed environment does
        nothing and just returns a list containing :attr:`data` itself.

    Note:
        Unlike PyTorch ``torch.distributed.all_gather_object``,
        :meth:`all_gather_object` in MMEngine does not pass in an empty list
        ``gather_list`` and returns the ``gather_list`` directly, which is
        more convenient. The difference between their interfaces is as below:

        - MMEngine: all_gather_object(data, group) -> gather_list
        - PyTorch: all_gather_object(gather_list, data, group) -> None

    Args:
        data (Any): Pickable Python object to be broadcast from current
            process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: Return a list containing data from the whole group if
        in distributed environment, otherwise a list only containing
        :attr:`data` itself.

    Note:
        For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]  # any picklable object
        >>> gather_objects = dist.all_gather_object(data[dist.get_rank()])
        >>> output
        ['foo']

        >>> # distributed environment
        >>> # We have 3 process groups, 3 ranks.
        >>> output = dist.all_gather_object(data[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]  # Rank 0
        ['foo', 12, {1: 2}]  # Rank 1
        ['foo', 12, {1: 2}]  # Rank 2
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    gather_list = [None] * world_size

    if digit_version(TORCH_VERSION) >= digit_version('1.8.0'):
        torch_dist.all_gather_object(gather_list, data, group)
    else:
        _all_gather_object(gather_list, data, group)

    return gather_list


def _object_to_tensor(obj: Any) -> Tuple[Tensor, Tensor]:
    """Serialize picklable python object to tensor."""
    byte_storage = torch.ByteStorage.from_buffer(pickle.dumps(obj))
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor
    # and specifying dtype. Otherwise, it will cause 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _tensor_to_object(tensor: Tensor, tensor_size: int) -> Any:
    """Deserialize tensor to picklable python object."""
    buf = tensor.cpu().numpy().tobytes()[:tensor_size]
    return pickle.loads(buf)
