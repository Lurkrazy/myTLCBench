import os
import argparse

import tvm
from tvm import relay, auto_scheduler

from utils import get_network, make_network_key

def allocate_task_time(tasks, task_weights, total_time, is_round_robin=True):
    """
    Allocate time for each task based on its weight or using round-robin method.
    
    Parameters:
    tasks (list): List of tasks
    task_weights (list): List of weights corresponding to each task
    total_time (int): Total available time
    is_round_robin (bool): Flag indicating whether to use round-robin method
    
    Returns:
    list: A list containing each task and its allocated time
    """
    if is_round_robin:
        # Round-robin allocation
        allocated_time = [total_time // len(tasks)] * len(tasks)
        remaining_time = total_time % len(tasks)
        for i in range(remaining_time):
            allocated_time[i] += 1
    else:
        total_weight = sum(task_weights)
        time_per_weight = total_time / total_weight
        # Generate the list of allocated times
        allocated_time = [round(weight * time_per_weight) for weight in task_weights]
    
    return allocated_time

def auto_scheduler_tune(network, batch_size, dtype, target, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)

    layout = "NCHW"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout
    )

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)
    
    total_time = 6*3600  # Total time in seconds
    allocated_task_times = allocate_task_time(tasks, task_weights, int(total_time), False)
    # print("Allocated task times:", allocated_task_times)
    # input("Press Enter to continue...")
    for task, allocated_time in zip(tasks, allocated_task_times):
        slide_window_size = 10  # Size of the sliding window used in dynamic gradient search
        max_tuning_time = allocated_time  # Maximum tuning time in seconds
        max_trials = 99999  # Maximum number of measurement trials to perform in dynamic gradient search
        n_start = 5  # Number of start points from the initial sampled population
        init_size = 64  # Number of samples to generate the initial model
        predict_score_threshold_ratio=0.6 # Threshold for the predict score
        measure_threshold_ratio=0.6 # Threshold for the measured throughput
        
        # Tuning options, tested with local runner and builder
        tune_option = auto_scheduler.TuningOptions(
            runner=auto_scheduler.LocalRunner(timeout=10),
            builder=auto_scheduler.LocalBuilder(timeout=10),
        )
        
        # initialize tuner
        tuner = auto_scheduler.dynamic_gradient_search.DynamicGradientSearchTuner(task, log_file, tune_option, n_start, 
                                                                                  init_size, slide_window_size, max_trials, max_tuning_time,
                                                                                  predict_score_threshold_ratio, measure_threshold_ratio)
        tuner.dynamic_gradient_search()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet_50", "mobilenet_v2", "bert", "all"],
        default="all",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=platinum-8124m -mcpu=skylake-avx512",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    parser.add_argument(
        "--logdir", type=str, default="tmp_logs/", help="Log file directory."
    )
    args = parser.parse_args()

    if args.network == "all":
        networks = ["resnet_50", "mobilenet_v2", "bert"]
    else:
        networks = [args.network]
    batch_sizes = [args.batch_size]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    for network in networks:
        for batch_size in batch_sizes:
            for dtype in dtypes:
                network_key = make_network_key(network, batch_size, dtype)
                print("Tune %s ..." % network_key)

                log_file = os.path.join(
                    args.logdir, "autoscheduler", target.model, network_key + ".json"
                )

                auto_scheduler_tune(network, batch_size, dtype, target, log_file)
