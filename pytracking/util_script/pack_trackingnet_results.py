import numpy as np
import os
import shutil
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.environment import env_settings
from pytracking.evaluation.trackingnetdataset import TrackingNetDataset

def pack_trackingnet_results(tracker_name, param_name, run_id=None, output_name=None):
    """ Packs trackingnet results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder env_settings().tn_packed_results_path

    args:
        tracker_name - name of the tracker
        param_name - name of the parameter file
        run_id - run id for the tracker
        output_name - name of the packed zip file
    """

    if output_name is None:
        if run_id is None:
            output_name = '{}_{}'.format(tracker_name, param_name)
        else:
            output_name = '{}_{}_{:03d}'.format(tracker_name, param_name, run_id)
    if env_settings().packed_results_path == '':
        raise RuntimeError('YOU HAVE NOT SETUP YOUR tn_packed_results_path in local.py!!!\n Go to "pytracking.evaluation.local" to set the path. '
                    'Then try to run again.')
    output_path = os.path.join(env_settings().packed_results_path, tracker_name, output_name)



    results_path = env_settings().results_path

    tn_dataset = TrackingNetDataset()

    for seq in tn_dataset:
        seq_name = seq.name
        if run_id is None:
            seq_results_path = '{}/{}/{}/{}.txt'.format(results_path, tracker_name, param_name, seq_name)
        else:
            seq_results_path = '{}/{}/{}_{:03d}/{}.txt'.format(results_path, tracker_name, param_name, run_id, seq_name)

        results = np.loadtxt(seq_results_path, dtype=np.float64)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savetxt('{}/{}.txt'.format(output_path, seq_name), results, delimiter=',', fmt='%.2f')

    # Generate ZIP file
    shutil.make_archive(output_path, 'zip', output_path)

    # Remove raw text files
    shutil.rmtree(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pack the trackinegnet results.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--output_name', type=str, default=None, help='The output zip file name.')

    args = parser.parse_args()
    pack_trackingnet_results(args.tracker_name, args.tracker_param, args.runid, args.output_name)
