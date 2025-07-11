"""
This module contains the logic for caching datasets.

import csv_dataset or json_dataset from here instead of inspect_ai.dataset if you want the data to be cached.
The functionality should be identical but the data will be cached on the first call and then loaded from file on subsequent calls.

The caches might build up over time, so you can delete the cached data by deleting the cached_data directory.
"""

import datetime
import hashlib
import inspect
import json
import pickle
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from inspect_ai.dataset import Dataset
from inspect_ai.dataset import csv_dataset as csv_dataset_base
from inspect_ai.dataset import json_dataset as json_dataset_base
from inspect_ai.dataset._util import shuffle_choices_if_requested

from .config import CACHE_DATA, DATA_DIR


def clean_for_serialization(obj: Any) -> Any:
    """Clean an object to make it json serializable."""
    if is_dataclass(obj):  # Needed to handle FieldSpec objects
        class_name = obj.__class__.__name__
        obj = asdict(obj)
        obj["__class__"] = class_name

    if callable(obj) and not isinstance(
        obj, (str, bytes, int, float, bool, type(None))
    ):
        # For functions, use their name and source code
        try:
            return f"function:{obj.__name__}:{inspect.getsource(obj)}"
        except (OSError, TypeError):
            # Fallback to function name and module if source is not available
            return f"function:{obj.__name__}:{getattr(obj, '__module__', 'unknown')}"

    if isinstance(obj, dict):
        return {k: clean_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        obj_ = [clean_for_serialization(item) for item in obj]
        obj = tuple(obj_) if isinstance(obj, tuple) else obj_
    return obj


def get_args_kwargs_str(*args, **kwargs) -> str:
    """
    Given the args and kwargs, return a string that can be used to identify the dataset and configs.

    Args:
        args: The arguments to the dataset function.
        kwargs: The keyword arguments to the dataset function.

    Returns:
        A string that can be used to identify the dataset and configs.
    """
    # Serialize args and kwargs, handling functions specially
    serialized_args = clean_for_serialization(args)
    serialized_kwargs = clean_for_serialization(kwargs)

    # Convert to JSON strings
    args_str = json.dumps(serialized_args, sort_keys=True)
    kwargs_str = json.dumps(serialized_kwargs, sort_keys=True)

    return args_str + kwargs_str


def get_dir_name(args_kwargs_str: str) -> str:
    """
    Hash the args and kwargs string and return the directory name to save the dataset to.

    Args:
        args_kwargs_str: The string that can be used to identify the dataset and configs.

    Returns:
        The directory name to save the dataset to.
    """
    hash_obj = hashlib.sha256(args_kwargs_str.encode())
    return f"{DATA_DIR}/{hash_obj.hexdigest()}"


def save_metadata(save_dir, metadata: dict):
    """
    Save the metadata to a json file.

    Args:
        save_dir: The directory to save the metadata to.
        metadata: The metadata to save.
    """
    with (save_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def load_from_cache(save_dir) -> Dataset:
    """
    Load the dataset from the cache.

    Update the last_accessed_at field in the metadata.

    Args:
        save_dir: The directory to load the dataset from.

    Returns:
        The dataset.
    """
    if not save_dir.exists():
        raise FileNotFoundError(f"Cache directory {save_dir} not found")

    # Load metadata
    with (save_dir / "metadata.json").open("r") as f:
        metadata = json.load(f)

    metadata["last_accessed_at"] = datetime.datetime.now().isoformat()
    save_metadata(save_dir, metadata)

    with (save_dir / "data.pkl").open("rb") as f:
        return pickle.load(f)


def save_to_cache(
    source_fn: Callable,
    dataset: Dataset,
    save_dir: Path,
    args_kwargs_str: str | None = None,
):
    """
    Save the dataset as pickle and metadata as json to the cache directory.

    Args:
        source_fn: The function that created the dataset.
        dataset: The dataset to save.
        save_dir: The directory to save the dataset to.
        args_kwargs_str: The string that can be used to identify the dataset and configs.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset
    with (save_dir / "data.pkl").open("wb") as f:
        pickle.dump(dataset, f)

    # Save metadata
    time_now = datetime.datetime.now().isoformat()
    metadata = {
        "created_at": time_now,
        "last_accessed_at": time_now,
        "num_samples": len(dataset),
        "name": dataset.name,
        "source_fn": source_fn.__name__,
        "location": dataset.location,
        "args_kwargs_str": args_kwargs_str,
    }
    save_metadata(save_dir, metadata)


def dataset_from_source(
    source_fn: Callable,
    *args,
    shuffle: bool = False,
    seed: int | None = None,
    shuffle_choices: bool | int | None = None,
    limit: int | None = None,
    **kwargs,
) -> Dataset:
    """
    A wrapper around the source function that will either load the dataset from the cache or create it and save it to the cache.

    Other than shuffle, seed, shuffle_choices and limit, all changes to args and kwargs will modify the hashing logic and will result in a
    fresh collection of the data and being saved to a new cache. Warning that if using a default kwarg, but then you call the function with that
    kwarg provided, even if it is the same as the default, it will be considered a different dataset and will be saved to a new cache.

    Args:
        source_fn: The function that creates the dataset either csv_dataset or json_dataset from inspect_ai.dataset
        args: The arguments to the source function.
        shuffle: Whether to shuffle the dataset.
        seed: The seed to use for shuffling.
        shuffle_choices: Whether to shuffle the choices.
        limit: The number of samples to limit the dataset to.
        kwargs: The keyword arguments to the source function.

    Returns:
        The dataset.
    """
    args_kwargs_str = get_args_kwargs_str(*args, **kwargs)
    save_dir = Path(get_dir_name(args_kwargs_str))

    try:
        # try to load the dataset from cache if it exists
        return load_from_cache(save_dir)
    except Exception:
        # I know this is general, but I want to catch all possible exceptions related to loading
        # the data or metadata, ok for now.
        pass

    # Call csv_dataset or json_dataset with the same arguments excluding those related to shuffling and sampling.
    # because we want to save all of the
    dataset = source_fn(
        *args,
        **kwargs,
    )
    save_to_cache(source_fn, dataset, save_dir, args_kwargs_str)

    ### logic from csv_dataset and json_dataset that was supressed in the above call.
    # shuffle if requested
    if shuffle:
        dataset.shuffle(seed=seed)

    shuffle_choices_if_requested(dataset, shuffle_choices)

    # limit if requested
    if limit:
        return dataset[0:limit]

    return dataset


def dataset_from_source_function_getter(
    source_fn: Callable,
) -> Callable:
    """
    A wrapper around _dataset_from_source

    Will either get the default version of the loader, or the version that handles caching, depending on the CACHE_DATA flag.

    Args:
        source_fn: The function to wrap with caching logic. either csv_dataset or json_dataset from inspect_ai.dataset

    Returns:
        The input function wrapped with caching logic, call it the same way as the source function.
    """
    if not CACHE_DATA:
        return source_fn

    return partial(dataset_from_source, source_fn)


csv_dataset = dataset_from_source_function_getter(csv_dataset_base)
json_dataset = dataset_from_source_function_getter(json_dataset_base)
