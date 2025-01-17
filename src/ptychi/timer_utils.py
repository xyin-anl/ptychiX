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
            print(func)
            if enabled and globals().get("ENABLE_TIMING", False):
                full_name = func.__qualname__
                remove_function_later = record_calling_function(func, full_name)

                torch.cuda.synchronize()
                start_time = time.time()
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                measure_overhead_start = time.time()
                if save_elapsed_time:
                    update_elapsed_time_array(func, full_name, elapsed_time)

                remove_calling_function(remove_function_later)

                # return result
            else:
                # If timing is disabled, just call the function
                result = func(*args, **kwargs)
            global TIMING_OVERHEAD_ARRAY
            TIMING_OVERHEAD_ARRAY = np.append(
                TIMING_OVERHEAD_ARRAY, time.time() - measure_overhead_start
            )
            # globals()["TIMING_OVERHEAD_ARRAY"] = np.append(
            #     globals()["TIMING_OVERHEAD_ARRAY"], time.time() - measure_overhead_start
            # )
            return result

        # Ensure the wrapper function has the same type as the original
        return wrapper  # type: ignore

    return decorator


def record_calling_function(func: callable, full_name: str) -> bool:
    # key = func.__qualname__
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
    key = func.__qualname__
    ELAPSED_TIME_DICT[key] = np.append(ELAPSED_TIME_DICT[key], elapsed_time)
    if key not in FUNCTION_CALL_STACK_DICT.keys():
        #     # FUNCTION_CALL_STACK_DICT[key] = get_calling_functions_from_module("ptychi")
        # FUNCTION_CALL_STACK_DICT[key] = get_calling_functions("ptychi")
        FUNCTION_CALL_STACK_DICT[key] = CALLING_FUNCTION_RUNNING_LIST.copy()


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


def get_calling_functions_from_module(
    module_name: str,
    exclude: list[str] = [
        "get_calling_functions_from_module",
        "update_elapsed_time_array",
        "wrapper",
    ],
) -> list[str]:
    """
    Get the functions from the specified module that called the current function.

    Args:
        module_name (str): Name of the module to filter functions by.

    Returns:
        list of str: List of function names from the module in the call stack.
    """

    calling_functions = []
    # Inspect the current stack
    for frame_info in inspect.stack():
        # Get the module where the function is defined
        module = inspect.getmodule(frame_info.frame)
        if (
            module
            and module.__name__.startswith(module_name)
            and frame_info.function not in exclude
        ):
            # Try to determine the class name
            class_name = None
            local_self = frame_info.frame.f_locals.get("self")
            local_cls = frame_info.frame.f_locals.get("cls")
            if local_self:  # If 'self' is in local variables, it's an instance method
                class_name = type(local_self).__name__
            elif local_cls:  # If 'cls' is in local variables, it's a class method
                class_name = local_cls.__name__

            # Construct the full name with module, class (if applicable), and function
            full_name = ""
            if class_name:
                full_name += f"{class_name}."
            full_name += frame_info.function
            calling_functions.append(full_name)

    return calling_functions


def get_calling_functions(include_module_name: str) -> list[str]:
    # Get the current stack
    stack = inspect.stack()
    calling_functions = []
    for frame_info in stack[16:17]:
        # Get module information
        module = inspect.getmodule(frame_info.frame)
        module_name = module.__name__ if module else "Unknown module"
        if include_module_name in module_name:
            # If the function is a method of a class, get the class name
            class_name = None
            if (
                "__class__" in frame_info.frame.f_locals
            ):  # issue is I don't know when this is even triggered
                class_name = frame_info.frame.f_locals["__class__"].__name__
            elif "self" in frame_info.frame.f_locals:
                class_name = frame_info.frame.f_locals["self"].__class__.__name__
            elif "cls" in frame_info.frame.f_locals:
                class_name = frame_info.frame.f_locals["cls"].__name__

            print((f"{class_name}.{frame_info.function}"))
            if class_name:
                calling_functions.append(f"{class_name}.{frame_info.function}")
            else:
                calling_functions.append(frame_info.function)
    return calling_functions


# # Get the current stack
# stack = inspect.stack()
# # print(f"Call stack for {func.__name__}:")
# i = 0
# for frame_info in stack:
#     # Get module information
#     module = inspect.getmodule(frame_info.frame)
#     module_name = module.__name__ if module else "Unknown module"
#     if "ptychi" in module_name:

#         # Get the class name if the function is a method of a class
#         class_name = None
#         # Get the first argument (`self` or `cls`) and check if it is an instance or class
#         # if "self" in frame_info.frame.f_locals:
#         #     class_name = frame_info.frame.f_locals["self"].__class__.__name__
#         #     # class_name = frame_info.frame.f_locals["__class__"].__name__
#         # elif "cls" in frame_info.frame.f_locals:
#         #     class_name = frame_info.frame.f_locals["cls"].__name__
#         #     print("p")

#         if "__class__" in frame_info.frame.f_locals:
#             class_name = frame_info.frame.f_locals["__class__"].__name__
#             # print("cls" in frame_info.frame.f_locals)
#         elif "self" in frame_info.frame.f_locals:
#             class_name = frame_info.frame.f_locals["self"].__class__.__name__
#         elif "cls" in frame_info.frame.f_locals:
#             class_name = frame_info.frame.f_locals["cls"].__name__

#         # print(f"{class_name}.{func.__name__}")

#         if class_name:
#             print(f"  {module_name}.{class_name}.{frame_info.function} (line {frame_info.lineno})")
#         else:
#             print(f"  {module_name}.{frame_info.function} (line {frame_info.lineno})")
#         # Construct the display string
#         # if class_name:
#         #     print(f"{class_name}.{frame_info.function}")
#         # else:
#         #     print(frame_info.function)

# # print("-" * 50)
