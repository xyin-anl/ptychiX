import time
from functools import wraps
from typing import Callable, TypeVar, Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from collections import defaultdict

# Type variables to retain function signatures
T = TypeVar("T", bound=Callable)

# Global flag to enable or disable timing
ENABLE_TIMING = True
ELAPSED_TIME_DICT = defaultdict(lambda: np.array([]))


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
                torch.cuda.synchronize()
                start_time = time.time()
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                if save_elapsed_time:
                    update_elapsed_time_array(prefix, func, elapsed_time)
                return result
            else:
                # If timing is disabled, just call the function
                return func(*args, **kwargs)

        # Ensure the wrapper function has the same type as the original
        return wrapper  # type: ignore

    return decorator


def update_elapsed_time_array(prefix: str, func: T, elapsed_time: float):
    key = prefix + func.__name__
    ELAPSED_TIME_DICT[key] = np.append(ELAPSED_TIME_DICT[key], elapsed_time)


def return_elapsed_time_arrays() -> dict:
    return ELAPSED_TIME_DICT


def delete_elapsed_time_arrays():
    global ELAPSED_TIME_DICT
    ELAPSED_TIME_DICT = defaultdict(lambda: np.array([]))


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
