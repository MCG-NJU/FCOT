import numpy as np
import multiprocessing
import os
from itertools import product
from pytracking.evaluation import Sequence, Tracker
import torch
multiprocessing.set_start_method('spawn', force=True)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, visdom_info=None, delimiter=None):
    """Runs a tracker on a sequence."""

    visdom_info = {} if visdom_info is None else visdom_info

    base_results_path = '{}/{}'.format(tracker.results_dir, seq.name)
    results_path = '{}.txt'.format(base_results_path)
    times_path = '{}_time.txt'.format(base_results_path)

    if os.path.isfile(results_path) and not debug:
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run(seq, debug=debug, visdom_info=visdom_info)
    else:
        try:
            output = tracker.run(seq, debug=debug, visdom_info=visdom_info)
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(output['target_bbox']).astype(int)
    exec_times = np.array(output['time']).astype(float)

    print('FPS: {}'.format(len(exec_times) / exec_times.sum()))
    if delimiter is None:
        delimiter = '\t'
    if not debug:
        np.savetxt(results_path, tracked_bb, delimiter=delimiter, fmt='%d')
        np.savetxt(times_path, exec_times, delimiter=delimiter, fmt='%f')


def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None, delimiter=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
    """
    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    visdom_info = {} if visdom_info is None else visdom_info

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, visdom_info=visdom_info, delimiter=delimiter)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, visdom_info, delimiter) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')


def run_vot(dataset, trackers, debug=False, threads=0, visdom_info=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
    """
    for tracker_info in trackers:
        tracker_info.run_vot(dataset, debug=debug, visdom_info=visdom_info)

    print('Done')