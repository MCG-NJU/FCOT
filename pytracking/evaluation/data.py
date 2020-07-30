import os
from glob import glob
from pytracking.evaluation.environment import env_settings


class BaseDataset:
    """Base class for all datasets."""
    def __init__(self):
        self.env_settings = env_settings()

    def __len__(self):
        """Overload this function in your dataset. This should return number of sequences in the dataset."""
        raise NotImplementedError

    def get_sequence_list(self):
        """Overload this in your dataset. Should return the list of sequences in the dataset."""
        raise NotImplementedError


class Sequence:
    """Class for the sequence in an evaluation."""
    def __init__(self, name, frames, ground_truth_rect):
        self.name = name
        self.frames = frames
        self.ground_truth_rect = ground_truth_rect
        self.env_settings = env_settings()
        self.pred_trajs = {}

    def init_info(self):
        return {key: self.get(key) for key in ['init_bbox']}

    def init_bbox(self):
        return list(self.ground_truth_rect[0,:])

    def get(self, name):
        return getattr(self, name)()

    def load_tracker(self, tracker_name, param_names, store=False):
        results_path = self.env_settings.results_path
        results_path = os.path.join(results_path, tracker_name)

        if isinstance(param_names, str):
            param_names = [param_names]
    
        for pname in param_names:
            pred_file = os.path.join(results_path, pname, '{}.txt'.format(self.name))
            if os.path.exists(pred_file):
                with open(pred_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split('\t')))
                            for x in f.readlines()]
                if len(pred_traj) != len(self.ground_truth_rect):
                    print(pname, len(pred_traj), len(self.ground_truth_rect), self.name)
                if store:
                    self.pred_trajs[pname] = pred_traj
                else:
                    return pred_traj
            else:
                print(pred_file)

                

class SequenceList(list):
    """List of sequences. Supports the addition operator to concatenate sequence lists."""
    def __getitem__(self, item):
        if isinstance(item, str):
            for seq in self:
                if seq.name == item:
                    return seq
            raise IndexError('Sequence name not in the dataset.')
        elif isinstance(item, int):
            return super(SequenceList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return SequenceList([super(SequenceList, self).__getitem__(i) for i in item])
        else:
            return SequenceList(super(SequenceList, self).__getitem__(item))

    def __add__(self, other):
        return SequenceList(super(SequenceList, self).__add__(other))

    def copy(self):
        return SequenceList(super(SequenceList, self).copy())