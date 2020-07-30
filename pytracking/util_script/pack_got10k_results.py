import os,shutil
import argparse
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.environment import env_settings


def main(tracker_name, param_name, run_id=None, output_name=None):

    if output_name is None:
        if run_id is None:
            output_name = '{}_{}'.format(tracker_name, param_name)
        else:
            output_name = '{}_{}_{:03d}'.format(tracker_name, param_name, run_id)
    if env_settings().packed_results_path == '':
        raise RuntimeError('YOU HAVE NOT SETUP YOUR tn_packed_results_path in local.py!!!\n Go to "pytracking.evaluation.local" to set the path. '
                    'Then try to run again.')
    output_path = os.path.join(env_settings().packed_results_path, tracker_name, output_name)
    got_results_dir = os.path.join(env_settings().packed_results_path, tracker_name, param_name)

    list = os.listdir(got_results_dir)
    for f in list:
        if f.split('.')[0][-4:] == 'time':
            print(f)
            prefix = f.split('.')[0][:-5]
            print(prefix)
            os.makedirs(os.path.join(got_results_dir, prefix))
            shutil.move(os.path.join(got_results_dir, f), os.path.join(got_results_dir,prefix,f))
            shutil.move(os.path.join(got_results_dir, prefix+'.txt'), os.path.join(got_results_dir, prefix, prefix+'.txt'))

    path = os.listdir(got_results_dir)
    for p in path:
        for file in os.listdir(os.path.join(got_results_dir, p)):
            if file.split('.')[0][-1] != 'e':
                os.rename(os.path.join(got_results_dir, p, file),
                          os.path.join(got_results_dir, p, file.split('.')[0] + '_001.txt'))

    shutil.make_archive(output_path, 'zip', got_results_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack the trackinegnet results.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--output_name', type=str, default=None, help='The output zip file name.')

    args = parser.parse_args()
    main(args.tracker_name, args.tracker_param, args.runid, args.output_name)