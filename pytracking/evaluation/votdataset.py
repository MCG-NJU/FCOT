import os
import numpy as np
from glob import glob
from PIL import Image
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


def VOTDataset():
    return VOTDatasetClass().get_sequence_list()


class VOTSequence(Sequence):
    """Class for the sequence in an evaluation."""
    def __init__(self, name, frames, ground_truth_rect):
        super().__init__(name, frames, ground_truth_rect)
        img = np.array(Image.open(frames[0]), np.uint8)
        self.width = img.shape[1]
        self.height = img.shape[0]

    def load_tracker(self, tracker_name, param_names, store=False):
        results_path = self.env_settings.results_path
        results_path = os.path.join(results_path, tracker_name)

        if isinstance(param_names, str):
            param_names = [param_names]
        for name in param_names:
            pred_files = glob(os.path.join(results_path, name, 'baseline', self.name, '*0*.txt'))
            if len(pred_files) == 15:
                pred_files = pred_files
            else:
                pred_files = pred_files[0:1]
            pred_traj = []
            for traj_file in pred_files:
                with open(traj_file, 'r') as f:
                    traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    pred_traj.append(traj)
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj
    
    def select_tag(self, tag, start=0, end=0):
        # if tag == 'empty':
        #     return self.tags[tag]
        return ([1]*len(self.ground_truth_rect))[start:end]


class VOTDatasetClass(BaseDataset):
    """VOT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        # if ground_truth_rect.shape[1] > 4:
        #     gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
        #     gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

        #     x1 = np.amin(gt_x_all, 1).reshape(-1,1)
        #     y1 = np.amin(gt_y_all, 1).reshape(-1,1)
        #     x2 = np.amax(gt_x_all, 1).reshape(-1,1)
        #     y2 = np.amax(gt_y_all, 1).reshape(-1,1)

        #     ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)

        return VOTSequence(sequence_name, frames, ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['ants1',
                        'ants3',
                        'bag',
                        'ball1',
                        'ball2',
                        'basketball',
                        'birds1',
                        'blanket',
                        'bmx',
                        'bolt1',
                        'bolt2',
                        'book',
                        'butterfly',
                        'car1',
                        'conduction1',
                        'crabs1',
                        'crossing',
                        'dinosaur',
                        'drone_across',
                        'drone_flip',
                        'drone1',
                        'fernando',
                        'fish1',
                        'fish2',
                        'fish3',
                        'flamingo1',
                        'frisbee',
                        'girl',
                        'glove',
                        'godfather',
                        'graduate',
                        'gymnastics1',
                        'gymnastics2',
                        'gymnastics3',
                        'hand',
                        'handball1',
                        'handball2',
                        'helicopter',
                        'iceskater1',
                        'iceskater2',
                        'leaves',
                        'matrix',
                        'motocross1',
                        'motocross2',
                        'nature',
                        'pedestrian1',
                        'rabbit',
                        'racing',
                        'road',
                        'shaking',
                        'sheep',
                        'singer2',
                        'singer3',
                        'soccer1',
                        'soccer2',
                        'soldier',
                        'tiger',
                        'traffic',
                        'wiper',
                        'zebrafish1']

        return sequence_list
