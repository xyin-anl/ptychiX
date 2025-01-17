import time
from functools import wraps
from typing import Callable, TypeVar, Optional, List, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from collections import defaultdict


# Type variables to retain function signatures
T = TypeVar("T", bound=Callable)

ENABLE_TIMING = True
"Global flag to enable or disable timing."
ELAPSED_TIME_DICT = defaultdict(lambda: np.array([]))
"""
A dictionary containing numpy arrays of the measured execution times of 
each timed function.
"""
ADVANCED_TIME_DICT = defaultdict(lambda: {})
"""
A nested dictionary, where each level of the dictionary contains
1) a key, value pair ("time": np.ndarray) that contains all measured
execution times for that function and 2) zero or more key-value 
pairs (function_name: dict) where function_name refers to the name
of each functions called in the function currently being times.

Note that only functions with the `timer` decorator will show up
in `ADVANCED_TIME_DICT`.
"""
CURRENT_DICT_REFERENCE = ADVANCED_TIME_DICT  # Initialized to the top level of ADVANCED_TIME_DICT
"""
A reference to the level of `ADVANCED_TIME_DICT` that corresponds to 
the function currently being executed.
"""
TIMING_OVERHEAD_ARRAY = np.array([])
"""
A numpy array containing measurements of how long it takes to execute 
functions in the `timer` decorator.
"""


def toggle_timer(enable: bool):
    global ENABLE_TIMING
    if enable is None:
        ENABLE_TIMING = not ENABLE_TIMING
    else:
        ENABLE_TIMING = enable


def timer(prefix: str = "", save_elapsed_time: bool = True, enabled: bool = True):
    """Decorator to time a function and print the function name if ENABLE_TIMING is True."""
    if prefix != "":
        prefix = prefix + "."

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled and globals().get("ENABLE_TIMING", False):
                # Measure the overhead from running the timer function
                measure_overhead_start_1 = time.time()
                full_name = func.__qualname__
                saved_dict_reference = update_current_dict_reference(full_name)
                overhead_time_1 = time.time() - measure_overhead_start_1

                # Measure function execution time
                torch.cuda.synchronize()
                start_time = time.time()
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time

                # Measure the overhead from running the timer function
                measure_overhead_start_2 = time.time()
                if save_elapsed_time:
                    update_elapsed_time_array(full_name, elapsed_time)
                    update_advanced_time_dict(elapsed_time)
                # Traverse back up the advanced timing dicts
                revert_current_dict_reference(saved_dict_reference)
                global TIMING_OVERHEAD_ARRAY
                overhead_time_2 = time.time() - measure_overhead_start_2
                TIMING_OVERHEAD_ARRAY = np.append(
                    TIMING_OVERHEAD_ARRAY,
                    overhead_time_1 + overhead_time_2,
                )
            else:
                # If timing is disabled, just call the function
                result = func(*args, **kwargs)
            # global TIMING_OVERHEAD_ARRAY
            # overhead_time_2 = time.time() - measure_overhead_start_2
            # TIMING_OVERHEAD_ARRAY = np.append(
            #     TIMING_OVERHEAD_ARRAY, overhead_time_1 + overhead_time_2
            # )
            return result

        # Ensure the wrapper function has the same type as the original
        return wrapper  # type: ignore

    return decorator


def update_elapsed_time_array(full_name: str, elapsed_time: float):
    ELAPSED_TIME_DICT[full_name] = np.append(ELAPSED_TIME_DICT[full_name], elapsed_time)


def update_current_dict_reference(full_name: str) -> dict:
    # Save the parent to traverse back to later
    saved_dict_reference = globals()["CURRENT_DICT_REFERENCE"]
    # Create new dict if necessary
    if full_name not in globals()["CURRENT_DICT_REFERENCE"].keys():
        globals()["CURRENT_DICT_REFERENCE"][full_name] = defaultdict(lambda: {})
        globals()["CURRENT_DICT_REFERENCE"][full_name]["time"] = np.array([])
    # Update the pointer to the current dict
    globals()["CURRENT_DICT_REFERENCE"] = globals()["CURRENT_DICT_REFERENCE"][full_name]
    return saved_dict_reference


def update_advanced_time_dict(elapsed_time: float):
    globals()["CURRENT_DICT_REFERENCE"]["time"] = np.append(
        globals()["CURRENT_DICT_REFERENCE"]["time"], elapsed_time
    )


def revert_current_dict_reference(saved_dict_reference):
    globals()["CURRENT_DICT_REFERENCE"] = saved_dict_reference


def return_elapsed_time_arrays() -> dict:
    return ELAPSED_TIME_DICT


def delete_elapsed_time_arrays():
    global ELAPSED_TIME_DICT
    global ADVANCED_TIME_DICT
    global CURRENT_DICT_REFERENCE
    global TIMING_OVERHEAD_ARRAY
    ELAPSED_TIME_DICT = defaultdict(lambda: np.array([]))
    ADVANCED_TIME_DICT = defaultdict(lambda: {})
    CURRENT_DICT_REFERENCE = ADVANCED_TIME_DICT
    TIMING_OVERHEAD_ARRAY = np.array([])


def plot_elapsed_time_bar_plot(
    elapsed_time_dict: dict,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    top_n: Optional[int] = None,
):
    elapsed_time_dict = return_dict_subset_copy(elapsed_time_dict, include, exclude)
    if top_n is not None:
        elapsed_time_dict = return_top_n_entries(elapsed_time_dict, top_n)

    # Remove the prefix from keys and calculate sums
    sums = [np.sum(elapsed_time_dict[key]) for key in elapsed_time_dict.keys()]

    # Sort the sums and corresponding keys in descending order
    sorted_indices = np.argsort(sums)[::-1]
    sorted_sums = [sums[i] for i in sorted_indices]
    sorted_keys = [list(elapsed_time_dict.keys())[i] for i in sorted_indices]

    generate_time_barplot(
        times=sorted_sums,
        labels=sorted_keys,
        colors="skyblue",
        title="Total elapsed time for each function",
    )


def plot_elapsed_time_bar_plot_advanced(
    data, key_to_find, max_levels=None, use_long_bar_labels: bool = False, figsize=None
):
    def find_key_in_nested_dict(nested_dict, target_key):
        for key, value in nested_dict.items():
            if key == target_key:
                return value
            if isinstance(value, dict):
                result = find_key_in_nested_dict(value, target_key)
                if result is not None:
                    return result
        return None

    # Find the dictionary containing the key
    result_dict = find_key_in_nested_dict(data, key_to_find)

    if result_dict is None:
        raise ValueError(f"Key '{key_to_find}' not found in the nested dictionary.")

    # Prepare data for plotting
    short_labels = []
    full_call_stack_labels = []
    times = []
    colors = []

    def collect_data(d, prefix="", function_name="", level=0):
        if max_levels is not None and level > max_levels:
            return
        symbol = " \u2192 "
        for sub_key, sub_value in d.items():
            if isinstance(sub_value, dict):
                collect_data(sub_value, prefix + sub_key + symbol, sub_key, level + 1)
            elif sub_key == "time":
                if sub_value.sum() < 1e-2:
                    return
                if prefix == "":
                    full_call_stack_labels.append("Total")
                    short_labels.append("Total")
                    colors.append("dodgerblue")
                else:
                    full_call_stack_labels.append(prefix.rstrip("/")[: -len(symbol)])
                    short_labels.append(function_name)
                    colors.append("skyblue")
                times.append(sub_value.sum())

    # Get total execution times for each of the called functions
    collect_data({k: v for k, v in result_dict.items() if k != "time"})

    # Add the total execution time of the function at the front of the list
    total_execution_time = result_dict["time"].sum()
    times.insert(0, total_execution_time)
    full_call_stack_labels.insert(0, "Total")
    short_labels.insert(0, "Total")
    colors.insert(0, "dodgerblue")

    if use_long_bar_labels:
        bar_labels = full_call_stack_labels
    else:
        bar_labels = short_labels

    generate_time_barplot(
        times=times,
        labels=bar_labels,
        colors=colors,
        title=f"Execution time breakdown for\n{key_to_find}",
    )

    print(text_bf(f"Execution summary of {key_to_find}\n"))
    print(text_bf(f"Total execution time: {total_execution_time:.3g} s\n"))
    print(text_bf("Execution times:"))
    for i in range(1, len(short_labels)):
        print(f"{i}. {short_labels[i]}: {times[i]:.2g} s")
    print()
    print(text_bf("Function call stack info:"))
    for i in range(1, len(full_call_stack_labels)):
        print(f"{i}. {full_call_stack_labels[i]}")

    selected_elapsed_times = dict(zip(short_labels, times))
    return selected_elapsed_times


def generate_time_barplot(
    times: dict,
    labels: list,
    colors: Optional[Union[list, str]],
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    label_fontsize: int = 10,
):
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(times)), times, color=colors)
    plt.gca().invert_yaxis()  # Invert the y-axis to have the largest bar on top
    plt.tick_params(left=False, labelleft=False)

    # Add labels at the start of each bar
    for i, (bar, label) in enumerate(zip(bars, labels)):
        plt.text(
            0,  # Start at the left edge
            bar.get_y() + bar.get_height() / 2,  # Vertically centered
            " " + label,
            ha="left",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
        )

    # Set x-axis and title
    plt.grid(linestyle=":")
    plt.gca().set_axisbelow(True)
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Function")
    plt.title(title)
    # plt.tight_layout()
    plt.show()


def plot_elapsed_time_vs_iteration(
    elapsed_time_dict: Dict[str, List[float]],
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    linestyle: str = "-",
    top_n: Optional[int] = None,  # plot top N largest sums
):
    elapsed_time_dict = return_dict_subset_copy(elapsed_time_dict, include, exclude)
    if top_n is not None:
        elapsed_time_dict = return_top_n_entries(elapsed_time_dict, top_n)

    for k, v in elapsed_time_dict.items():
        if hasattr(v, "__len__") and len(v) > 2:  # temp fix
            plt.plot(v, linestyle, label=k)

    plt.legend()
    plt.grid(linestyle=":")
    plt.gca().set_axisbelow(True)
    plt.ylabel("Elapsed Time (s)")
    plt.xlabel("Iteration")
    plt.title("Elapsed time vs iteration")
    plt.tight_layout()
    plt.show()


def return_dict_subset_copy(
    elapsed_time_dict: dict,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> dict:
    elapsed_time_dict = copy.deepcopy(elapsed_time_dict)
    if exclude is not None:
        [elapsed_time_dict.pop(k) for k in exclude]
    if include is not None:
        elapsed_time_dict = {
            k: elapsed_time_dict[k] for k in elapsed_time_dict.keys() if k in include
        }
    return elapsed_time_dict


def return_top_n_entries(elapsed_time_dict: dict, top_n: int) -> dict:
    elapsed_time_dict = copy.deepcopy(elapsed_time_dict)
    # if top_n is not None:
    # Sort dictionary items based on the sum of their values and get top N
    sorted_items = sorted(
        elapsed_time_dict.items(),
        # key=lambda x: sum(x[1]) if hasattr(x[1], "__iter__") else float("-inf"),
        key=lambda x: sum(x[1]) if hasattr(x[1], "__iter__") else x[1],
        reverse=True,
    )[:top_n]
    elapsed_time_dict = dict(sorted_items)

    return elapsed_time_dict


def text_bf(string: str):
    return f"\033[1m{string}\033[0m"
