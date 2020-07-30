import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.otbdataset import OTBDataset
from pytracking.evaluation.nfsdataset import NFSDataset
from pytracking.evaluation.uavdataset import UAVDataset
from pytracking.evaluation.votdataset import VOTDataset
from pytracking.evaluation.lasotdataset import LaSOTDataset
from pytracking.evaluation.trackingnetdataset import TrackingNetDataset
from pytracking.evaluation.got10kdataset import GOT10KDatasetTest, GOT10KDatasetVal, GOT10KDatasetLTRVal
from pytracking.evaluation.running import run_dataset, run_vot
from pytracking.evaluation import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                visdom_info=None, delimiter=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    if dataset_name == 'otb':
        dataset = OTBDataset()
    elif dataset_name == 'nfs':
        dataset = NFSDataset()
    elif dataset_name == 'uav':
        dataset = UAVDataset()
    elif dataset_name == 'vot':
        dataset = VOTDataset()
    elif dataset_name == 'tn':
        dataset = TrackingNetDataset()
    elif dataset_name == 'gott':
        dataset = GOT10KDatasetTest()
    elif dataset_name == 'gotv':
        dataset = GOT10KDatasetVal()
    elif dataset_name == 'gotlv':
        dataset = GOT10KDatasetLTRVal()
    elif dataset_name == 'lasot':
        dataset = LaSOTDataset()
    else:
        raise ValueError('Unknown dataset name')

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    if dataset_name != 'vot':
        run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info)
    else:
        run_vot(dataset, trackers, debug, threads, visdom_info=visdom_info)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=str, default=None, help='The run id.')
    parser.add_argument('--dataset', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
    parser.add_argument('--delimiter', type=str, default='\t', help='Type for saving tracking results.')
    parser.add_argument('--cuda_id', type=str, default='0', help='Specific cuda to use for running tracker.')

    args = parser.parse_args()

    # set cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset, args.sequence, args.debug, args.threads,
                {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, args.delimiter)


if __name__ == '__main__':
    main()
