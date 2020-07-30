import os
import sys
import time
import argparse
import functools

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
# from pysot.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset

from pytracking.benchmarks import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark#, F1Benchmark
from pytracking.evaluation.otbdataset import OTBDataset
from pytracking.evaluation.nfsdataset import NFSDataset
from pytracking.evaluation.uavdataset import UAVDataset
from pytracking.evaluation.votdataset import VOTDataset
from pytracking.evaluation.lasotdataset import LaSOTDataset

from pytracking.visualization import draw_success_precision, draw_eao, draw_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_params', nargs='+', help='Names of parameters.')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--tracker_result_root', type=str, help='tracker result root')
    parser.add_argument('--result_dirs', nargs='+')
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--show_video_level', dest='show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=1)
    args = parser.parse_args()

    tracker_name = args.tracker_name
    tracker_params = args.tracker_params
    # root = args.dataset_dir

    assert len(tracker_params) > 0
    args.num = min(args.num, len(tracker_params))

    if 'OTB' in args.dataset:
        # dataset = OTBDataset(args.dataset, root)
        # dataset.set_tracker(tracker_dir, tracker_params)
        dataset = OTBDataset()
        benchmark = OPEBenchmark(dataset, tracker_name, tracker_params)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                tracker_params), desc='eval success', total=len(tracker_params), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                tracker_params), desc='eval precision', total=len(tracker_params), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            # for attr, videos in dataset.attr.items():
            video_list = [seq.name for seq in dataset]
            draw_success_precision(success_ret,
                        name=args.dataset,
                        videos=video_list,
                        attr='ALL',
                        precision_ret=precision_ret)
    elif 'LaSOT' == args.dataset:
        # dataset = LaSOTDataset(args.dataset, root)
        # dataset.set_tracker(tracker_dir, trackers)
        dataset = LaSOTDataset()
        benchmark = OPEBenchmark(dataset, tracker_name, tracker_params)
        success_ret = {}
        # success_ret = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                tracker_params), desc='eval success', total=len(tracker_params), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                tracker_params), desc='eval precision', total=len(tracker_params), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                tracker_params), desc='eval norm precision', total=len(tracker_params), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            video_list = [seq.name for seq in dataset]
            draw_success_precision(success_ret,
                    name=args.dataset,
                    videos=video_list,
                    attr='ALL',
                    precision_ret=precision_ret,
                    norm_precision_ret=norm_precision_ret)
    elif 'UAV' in args.dataset:
        # dataset = UAVDataset(args.dataset, root)
        # dataset.set_tracker(tracker_dir, trackers)
        dataset = UAVDataset()
        benchmark = OPEBenchmark(dataset, tracker_name, tracker_params)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                tracker_params), desc='eval success', total=len(tracker_params), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                tracker_params), desc='eval precision', total=len(tracker_params), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            # for attr, videos in dataset.attr.items():
            video_list = [seq.name for seq in dataset]
            draw_success_precision(success_ret,
                    name=args.dataset,
                    videos=video_list,
                    attr='ALL',
                    precision_ret=precision_ret)
    elif 'NFS' in args.dataset:
        # dataset = NFSDataset(args.dataset, root)
        # dataset.set_tracker(tracker_dir, trackers)
        dataset = NFSDataset()
        benchmark = OPEBenchmark(dataset, tracker_name, tracker_params)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                tracker_params), desc='eval success', total=len(tracker_params), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                tracker_params), desc='eval precision', total=len(tracker_params), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            # for attr, videos in dataset.attr.items():
            video_list = [seq.name for seq in dataset]
            draw_success_precision(success_ret,
                        name=args.dataset,
                        video=video_list,
                        attr='ALL',
                        precision_ret=precision_ret)
    elif 'VOT2018' == args.dataset:
        # dataset = VOTDataset(args.dataset, root)
        # dataset.set_tracker(tracker_dir, trackers)
        dataset = VOTDataset()
        ar_benchmark = AccuracyRobustnessBenchmark(dataset, tracker_name, tracker_params)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                tracker_params), desc='eval ar', total=len(tracker_params), ncols=100):
                ar_result.update(ret)
        # benchmark.show_result(ar_result)

        benchmark = EAOBenchmark(dataset, tracker_name, tracker_params)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                tracker_params), desc='eval eao', total=len(tracker_params), ncols=100):
                eao_result.update(ret)
        # benchmark.show_result(eao_result)
        ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=args.show_video_level)
    else:
        print('Not support dataset {}, please input again.'.format(args.dataset))
    # elif 'VOT2018-LT' == args.dataset:
    #     dataset = VOTLTDataset(args.dataset, root)
    #     dataset.set_tracker(tracker_dir, trackers)
    #     benchmark = F1Benchmark(dataset)
    #     f1_result = {}
    #     with Pool(processes=args.num) as pool:
    #         for ret in tqdm(pool.imap_unordered(benchmark.eval,
    #             trackers), desc='eval f1', total=len(trackers), ncols=100):
    #             f1_result.update(ret)
    #     benchmark.show_result(f1_result,
    #             show_video_level=args.show_video_level)
    #     if args.vis:
    #         draw_f1(f1_result)
