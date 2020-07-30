import importlib
import os
import cv2
import numpy as np
import pickle
from pytracking.evaluation.environment import env_settings
from pytracking.utils.bbox import get_axis_aligned_bbox
from pytracking.utils.vot_utils.region import vot_overlap, vot_float2str


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None):
        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
        self.tracker_class = tracker_module.get_tracker_class()
        self.params = self.get_parameters()


    def run(self, seq, visualization=None, debug=None, visdom_info=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
        """
        visdom_info = {} if visdom_info is None else visdom_info
        # params = self.get_parameters()
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(self.params, 'debug', 0)
        if visualization is None:
            if debug is None:
                visualization_ = getattr(self.params, 'visualization', False)
            else:
                visualization_ = True if debug else False

        self.params.visualization = visualization_
        self.params.debug = debug_
        self.params.visdom_info = visdom_info

        tracker = self.tracker_class(self.params)

        output = tracker.track_sequence(seq)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        visdom_info = {} if visdom_info is None else visdom_info

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        params.visdom_info = visdom_info

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        tracker = self.tracker_class(params)
        tracker.track_videofile(videofilepath, optional_box)

    def run_webcam(self, debug=None, visdom_info=None):
        """Run the tracker with the webcam.
        args:
            debug: Debug level.
        """
        visdom_info = {} if visdom_info is None else visdom_info
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.visdom_info = visdom_info

        tracker = self.tracker_class(params)

        tracker.track_webcam()

    def run_vot(self, dataset, debug=None, visdom_info=None):
        """ Run on vot"""
        visdom_info = {} if visdom_info is None else visdom_info
        # self.params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(self.params, 'debug', 0)
        self.params.debug = debug_

        self.params.tracker_name = self.name
        self.params.param_name = self.parameter_name
        self.params.run_id = self.run_id
        self.params.visdom_info = visdom_info

        tracker = self.tracker_class(self.params)
        # tracker.initialize_features()
        # tracker.track_vot()
        total_lost = 0

        for v_idx, seq in enumerate(dataset):

            frame_counter = 0
            lost_number = 0
            toc = 0

            seq_name = seq.name
            seq_frames_path = seq.frames
            seq_ground_truth_rect = seq.ground_truth_rect

            pred_bboxes = []
            # for idx, (img, gt_bbox) in enumerate(seq_list):
            for idx, img_path in enumerate(seq_frames_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_bbox = seq_ground_truth_rect[idx]
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                    gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                    gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                    gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    out = tracker.initialize(img, {'init_bbox': gt_bbox_})
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    out = tracker.track(img)
                    pred_bbox = out['target_bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))

                    if overlap > 0:
                    # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                    # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()

            toc /= cv2.getTickFrequency()
        # save results
            video_path = os.path.join(self.results_dir,
                'baseline', seq_name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(seq_name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx+1, seq_name, toc, idx / toc, lost_number))
            total_lost += lost_number
            print('total lost: {}'.format(total_lost))
        print("{:s} total lost: {:d}".format(self.name, total_lost))

        
    def get_parameters(self):
        """Get parameters."""

        parameter_file = '{}/parameters.pkl'.format(self.results_dir)
        if os.path.isfile(parameter_file):
            return pickle.load(open(parameter_file, 'rb'))

        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()

        if self.run_id is not None:
            pickle.dump(params, open(parameter_file, 'wb'))

        return params


