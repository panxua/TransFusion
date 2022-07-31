from unittest import result
import mmcv
import numpy as np

from mmdet.datasets import DATASETS, CustomDataset

from mmdet3d.datasets.kitti2d_dataset import Kitti2DDataset


@DATASETS.register_module()
class VOD2DDataset(Kitti2DDataset):
    r"""VOD 2D Dataset.

    This class serves as the API for experiments on the `VOD Dataset
    <http://www.cvlibs.net/datasets/VOD/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    CLASSES = ('car', 'pedestrian', 'cyclist')
    """
    Annotation format:
    [
        {
            'image': {
                'image_idx': 0,
                'image_path': 'training/image_2/000000.png',
                'image_shape': array([ 370, 1224], dtype=int32)
            },
            'point_cloud': {
                 'num_features': 4,
                 'velodyne_path': 'training/velodyne/000000.bin'
             },
             'calib': {
                 'P0': <np.ndarray> (4, 4),
                 'P1': <np.ndarray> (4, 4),
                 'P2': <np.ndarray> (4, 4),
                 'P3': <np.ndarray> (4, 4),
                 'R0_rect':4x4 np.array,
                 'Tr_velo_to_cam': 4x4 np.array,
                 'Tr_imu_to_velo': 4x4 np.array
             },
             'annos': {
                 'name': <np.ndarray> (n),
                 'truncated': <np.ndarray> (n),
                 'occluded': <np.ndarray> (n),
                 'alpha': <np.ndarray> (n),
                 'bbox': <np.ndarray> (n, 4),
                 'dimensions': <np.ndarray> (n, 3),
                 'location': <np.ndarray> (n, 3),
                 'rotation_y': <np.ndarray> (n),
                 'score': <np.ndarray> (n),
                 'index': array([0], dtype=int32),
                 'group_ids': array([0], dtype=int32),
                 'difficulty': array([0], dtype=int32),
                 'num_points_in_gt': <np.ndarray> (n),
             }
        }
    ]
    """

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        self.data_infos = mmcv.load(ann_file)
        self.cat2label = {
            cat_name: i
            for i, cat_name in enumerate(self.CLASSES)
        }
        return self.data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if len(img_info['annos']['name']) > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]['image']
            if img_info['image_shape'][1] / img_info['image_shape'][0] > 1:
                self.flag[i] = 1

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - bboxes (np.ndarray): Ground truth bboxes.
                - labels (np.ndarray): Labels of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        annos = info['annos']
        gt_names = annos['name']
        gt_bboxes = annos['bbox']
        difficulty = annos['difficulty']

        # remove classes that is not needed
        selected = self.keep_arrays_by_name(gt_names, self.CLASSES)
        gt_bboxes = gt_bboxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        gt_labels = np.array([self.cat2label[n] for n in gt_names])

        anns_results = dict(
            bboxes=gt_bboxes.astype(np.float32),
            labels=gt_labels,
        )
        return anns_results

    def prepare_train_img(self, idx):
        """Training image preparation.

        Args:
            index (int): Index for accessing the target image data.

        Returns:
            dict: Training image data dict after preprocessing
                corresponding to the index.
        """
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        ann_info = self.get_ann_info(idx)
        if len(ann_info['bboxes']) == 0:
            return None

        def filter_nonzero(entry):
            w,h = entry[0][2]-entry[0][0], entry[0][3]-entry[0][1]
            return w>0 and h>0
        filtered_ann_info = list(filter(filter_nonzero, zip(ann_info['bboxes'],ann_info['labels'])))
        if len(filtered_ann_info)==0:
            return None
        # if(len(filtered_ann_info) < len(ann_info['bboxes'])):
        #     print()
        ann_info['bboxes'], ann_info['labels'] = zip(*filtered_ann_info)
        ann_info['bboxes'], ann_info['labels'] = np.array(ann_info['bboxes']), np.array(ann_info['labels'])

        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target image data.

        Returns:
            dict: Testing image data dict after preprocessing
                corresponding to the index.
        """
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def reformat_bbox(self, outputs, out=None):
        """Reformat bounding boxes to VOD 2D styles.

        Args:
            outputs (list[np.ndarray]): List of arrays storing the inferenced
                bounding boxes and scores.
            out (str | None): The prefix of output file. Default: None.

        Returns:
            list[dict]: A list of dictionaries with the VOD 2D format.
        """
        from mmdet3d.core.bbox.transforms import bbox2result_VOD2d
        sample_idx = [info['image']['image_idx'] for info in self.data_infos]
        result_files = bbox2result_VOD2d(outputs, self.CLASSES, sample_idx,
                                           out)
        return result_files

    def evaluate(self, result_files, logger=None, eval_types=None):
        """Evaluation in VOD protocol.

        Args:
            result_files (str): Path of result files.
            eval_types (str): Types of evaluation. Default: None.
                VOD dataset only support 'bbox' evaluation type.

        Returns:
            tuple (str, dict): Average precision results in str format
                and average precision results in dict format.
        """

        from mmdet3d.core.evaluation import kitti_eval
        eval_types = ['bbox'] if not eval_types else eval_types
        assert eval_types in ('bbox', ['bbox'
                                       ]), 'KITTI data set only evaluate bbox'
        gt_annos = [info['annos'] for info in self.data_infos]

        dt_annos = []
        for anno in result_files:
            dt_name, dt_bbox, dt_score = [],[],[]
            for class_id, bbox in enumerate(anno):
                dt_name += [self.CLASSES[class_id]]*len(bbox)
                dt_bbox += list(bbox[:,:4])
                dt_score += list(bbox[:,4])
            dt_anno = {
                "name":np.array(dt_name),
                "bbox":np.array(dt_bbox),
                "alpha":np.array([]),
                "score":np.array(dt_score)
                }
            dt_annos.append(dt_anno)


        ap_result_str, ap_dict = kitti_eval(
            gt_annos, dt_annos, self.CLASSES, eval_types=['bbox'])
        return ap_dict

        # gt_annos = [info['annos'] for info in self.data_infos]

        # from mmdet3d.core.evaluation.vod_utils import evaluate
        # if isinstance(result_files, dict):
        #     dt_annos = result_files['pts_bbox']
        # eval_results = evaluate(gt_annos,dt_annos)
        # ap_dict = {"entire overall": (eval_results['entire_area']['Car_3d_all'] + eval_results['entire_area']['Pedestrian_3d_all'] + eval_results['entire_area']['Cyclist_3d_all']) / 3,
        #             "corridor overall":(eval_results['roi']['Car_3d_all'] + eval_results['roi']['Pedestrian_3d_all'] + eval_results['roi']['Cyclist_3d_all']) / 3}
        # for k1, temp_dict in eval_results.items():
        #     for k2, value in temp_dict.items():
        #         ap_dict["{} {}".format(k1,k2)] = value
        # result = ("Results: \n"
        #         f"Entire annotated area: \n"
        #         f"Car: {eval_results['entire_area']['Car_3d_all']} \n"
        #         f"Pedestrian: {eval_results['entire_area']['Pedestrian_3d_all']} \n"
        #         f"Cyclist: {eval_results['entire_area']['Cyclist_3d_all']} \n"
        #         f"mAP: {(eval_results['entire_area']['Car_3d_all'] + eval_results['entire_area']['Pedestrian_3d_all'] + eval_results['entire_area']['Cyclist_3d_all']) / 3} \n"
        #         f"Driving corridor area: \n"
        #         f"Car: {eval_results['roi']['Car_3d_all']} \n"
        #         f"Pedestrian: {eval_results['roi']['Pedestrian_3d_all']} \n"
        #         f"Cyclist: {eval_results['roi']['Cyclist_3d_all']} \n"
        #         f"mAP: {(eval_results['roi']['Car_3d_all'] + eval_results['roi']['Pedestrian_3d_all'] + eval_results['roi']['Cyclist_3d_all']) / 3} \n"
        #     )
        # return result, ap_dict