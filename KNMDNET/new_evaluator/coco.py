# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path as osp
import pickle
import uuid
import new_evaluator.ds_utils as ds_utils
import numpy as np
import scipy.sparse
from new_evaluator.imdb import imdb
from setting.read_setting import config as cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
class coco(imdb):
    def __init__(self, save_dir,image_set, year):
        imdb.__init__(self, 'coco_' + year + '_' + image_set)
        # COCO specific config options
        self.config = {'use_salt': True,
                       'cleanup': True}
        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg["coco_data_dir"], 'COCO')
        # load COCO API, classes, class <-> id mappings
        self._COCO = COCO(self._get_ann_file())
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                                   self._COCO.getCatIds())))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self.set_proposal_method('gt')
        self.competition_mode(False)
        self.save_dir=save_dir
        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.
        self._view_map = {
            'minival2014': 'val2014',  # 5k val2014 subset
            'valminusminival2014': 'val2014',  # val2014 \setminus minival2014
            'test-dev2015': 'test2015',
        }
        coco_name = image_set + year  # e.g., "val2014"
        self._data_name = (self._view_map[coco_name]
                           if coco_name in self._view_map
                           else coco_name)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('train', 'val', 'minival')

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 \
            else 'image_info'
        return osp.join(self._data_path, 'Annotations',self._image_set+ '.json')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._COCO.getImgIds()
        return image_ids

    def _get_widths(self):
        anns = self._COCO.loadImgs(self._image_index)
        widths = [ann['width'] for ann in anns]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = ('COCO_' + self._data_name + '_' +
                     str(index).zfill(12) + '.jpg')
        image_path = osp.join(self._data_path, 'images',
                              self._data_name, file_name)
        assert osp.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_coco_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _load_coco_annotation(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        ds_utils.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'width': width,
                'height': height,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_widths(self):
        return [r['width'] for r in self.roidb]

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'width': widths[i],
                     'height': self.roidb[i]['height'],
                     'boxes': boxes,
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'flipped': True,
                     'seg_areas': self.roidb[i]['seg_areas']}

            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def _get_box_file(self, index):
        file_name = ('COCO_' + self._data_name +
                     '_' + str(index).zfill(12) + '.mat')
        return osp.join(file_name[:14], file_name[:22], file_name)

    def _print_detection_eval_metrics(self, coco_eval):
        stats = [[]for _ in range(12)]
        precision = coco_eval.eval['precision'][0, :, 0, 0, 2]

        maxDets=100
        stats[0] = self.eval_summarize(coco_eval=coco_eval,ap=1)
        stats[1] = self.eval_summarize(coco_eval=coco_eval,ap=1, iouThr=.5, maxDets=maxDets)
        stats[2] = self.eval_summarize(coco_eval=coco_eval,ap=1, iouThr=.75, maxDets=maxDets)
        stats[3] = self.eval_summarize(coco_eval=coco_eval,ap=1, iouThr=.5,areaRng='small', maxDets=maxDets)
        stats[4] = self.eval_summarize(coco_eval=coco_eval,ap=1, iouThr=.5,areaRng='medium', maxDets=maxDets)
        stats[5] = self.eval_summarize(coco_eval=coco_eval,ap=1, iouThr=.5,areaRng='large', maxDets=maxDets)
        stats[6] = self.eval_summarize(coco_eval=coco_eval,ap=0)
        stats[7] = self.eval_summarize(coco_eval=coco_eval,ap=0, iouThr=.5, maxDets=maxDets)
        stats[8] = self.eval_summarize(coco_eval=coco_eval,ap=0, iouThr=.75, maxDets=maxDets)
        stats[9] = self.eval_summarize(coco_eval=coco_eval,ap=0, iouThr=.5,areaRng='small', maxDets=maxDets)
        stats[10] = self.eval_summarize(coco_eval=coco_eval,ap=0, iouThr=.5,areaRng='medium', maxDets=maxDets)
        stats[11] = self.eval_summarize(coco_eval=coco_eval,ap=0, iouThr=.5,areaRng='large', maxDets=maxDets)
        return stats,precision
    def eval_summarize(self,coco_eval,ap=1, iouThr=None, areaRng='all',maxDets=100):
        p = coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        name=typeStr+'@'+iouStr+areaRng
        return name,mean_s
    def _do_train_eval(self,
                       res_file,
                       output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 16 ** 2], [16 ** 2, 23 ** 2], [23 ** 2, 1e5 ** 2]]
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        stats, precision = self._print_detection_eval_metrics(coco_eval)
    def _do_detection_eval(self,
                           res_file,
                           output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 16 ** 2], [16 ** 2, 23 ** 2], [23 ** 2, 1e5 ** 2]]
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        stats,precision=self._print_detection_eval_metrics(coco_eval)
        eval_file = osp.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))
        return stats,precision
    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(float)
            if len(dets) == 0:
                continue
            dets[:,1:]=dets[:,1:]
            scores = dets[:, 0]
            xs = dets[:, 1]
            ys = dets[:, 2]
            ws = dets[:, 3] - xs + 1
            hs = dets[:, 4] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,self.num_classes - 1))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind-1],coco_cat_id))

        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = osp.join(output_dir, ('detections_' +
                                         self._image_set +
                                         self._year +
                                         '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
