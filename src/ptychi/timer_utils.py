import time
from functools import wraps
from typing import Callable, TypeVar, Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from collections import defaultdict
import inspect


# Type variables to retain function signatures
T = TypeVar("T", bound=Callable)

# Global flag to enable or disable timing
ENABLE_TIMING = True
ELAPSED_TIME_DICT = defaultdict(lambda: np.array([]))
FUNCTION_CALL_STACK_DICT = defaultdict(list)
TIMING_OVERHEAD_ARRAY = np.array([])
CALLING_FUNCTION_RUNNING_LIST = []  # list[str] # Running list of functions that have been called

ADVANCED_TIME_DICT = defaultdict(lambda: {})
CURRENT_DICT = ADVANCED_TIME_DICT  # Points to the current dict


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
            # print(func)
            if enabled and globals().get("ENABLE_TIMING", False):
                measure_overhead_start_1 = time.time()
                full_name = func.__qualname__
                remove_function_later = record_calling_function(func, full_name)
                parent_dict = update_current_dict_pointer(func, full_name)
                overhead_time_1 = time.time() - measure_overhead_start_1

                torch.cuda.synchronize()
                start_time = time.time()
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                measure_overhead_start_2 = time.time()
                if save_elapsed_time:
                    update_elapsed_time_array(func, full_name, elapsed_time)
                    update_advanced_time_dict(func, full_name, elapsed_time)

                remove_calling_function(remove_function_later)

                # Traverse back up the advanced timing dicts
                move_current_dict_back(parent_dict)

            else:
                # If timing is disabled, just call the function
                result = func(*args, **kwargs)
            global TIMING_OVERHEAD_ARRAY
            overhead_time_2 = time.time() - measure_overhead_start_2
            TIMING_OVERHEAD_ARRAY = np.append(
                TIMING_OVERHEAD_ARRAY, overhead_time_1 + overhead_time_2
            )
            return result

        # Ensure the wrapper function has the same type as the original
        return wrapper  # type: ignore

    return decorator


def record_calling_function(func: callable, full_name: str) -> bool:
    if full_name not in ELAPSED_TIME_DICT.keys():
        globals()["CALLING_FUNCTION_RUNNING_LIST"] += [full_name]
        remove_function_later = True
    else:
        remove_function_later = False
    return remove_function_later


def remove_calling_function(remove_function_from_list: bool):
    if remove_function_from_list:
        del CALLING_FUNCTION_RUNNING_LIST[-1]


def update_elapsed_time_array(func: T, full_name: str, elapsed_time: float):
    ELAPSED_TIME_DICT[full_name] = np.append(ELAPSED_TIME_DICT[full_name], elapsed_time)
    if full_name not in FUNCTION_CALL_STACK_DICT.keys():
        FUNCTION_CALL_STACK_DICT[full_name] = CALLING_FUNCTION_RUNNING_LIST.copy()


def update_current_dict_pointer(func: T, full_name: str) -> dict:
    # Save the parent to traverse back to later
    parent_dict = globals()["CURRENT_DICT"]
    # Create new dict if necessary
    if full_name not in globals()["CURRENT_DICT"].keys():
        globals()["CURRENT_DICT"][full_name] = defaultdict(lambda: {})
        globals()["CURRENT_DICT"][full_name]["time"] = 0
    # Update the pointer to the current dict
    globals()["CURRENT_DICT"] = globals()["CURRENT_DICT"][full_name]
    return parent_dict


def update_advanced_time_dict(func: T, full_name: str, elapsed_time: float):
    globals()["CURRENT_DICT"]["time"] += elapsed_time


def move_current_dict_back(parent_dict):
    globals()["CURRENT_DICT"] = parent_dict


def return_elapsed_time_arrays() -> dict:
    return ELAPSED_TIME_DICT


def return_call_stack_dict() -> dict:
    return FUNCTION_CALL_STACK_DICT


def delete_elapsed_time_arrays():
    global ELAPSED_TIME_DICT
    global FUNCTION_CALL_STACK_DICT
    ELAPSED_TIME_DICT = defaultdict(lambda: np.array([]))
    FUNCTION_CALL_STACK_DICT = defaultdict(list)


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

    # Create a horizontal bar plot
    plt.figure()  # figsize=(10, 6))
    bars = plt.barh(range(len(sorted_sums)), sorted_sums, color="skyblue")
    plt.gca().invert_yaxis()  # Invert the y-axis to have the largest bar on top
    plt.tick_params(left=False, labelleft=False)

    # Add labels at the start of each bar
    for i, (bar, label) in enumerate(zip(bars, sorted_keys)):
        plt.text(
            0,  # Start at the left edge
            bar.get_y() + bar.get_height() / 2,  # Vertically centered
            " " + label,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Set x-axis and title
    plt.grid(linestyle=":")
    plt.gca().set_axisbelow(True)
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Function")
    plt.title("Total elapsed time for each function")
    plt.tight_layout()
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
    # plt.show()


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
