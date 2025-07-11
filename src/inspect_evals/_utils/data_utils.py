import datetime
import hashlib
import inspect
import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable

from inspect_ai.dataset import Dataset
from inspect_ai.dataset import csv_dataset as csv_dataset_base
from inspect_ai.dataset import json_dataset as json_dataset_base
from inspect_ai.dataset._util import shuffle_choices_if_requested

DATA_DIR = Path(__file__).parent.parent.parent.parent / "cached_data"


def serialize_for_hashing(obj) -> str:
    """Serialize an object for hashing, handling functions specially."""
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
    elif isinstance(obj, (dict, list, tuple)):
        # Recursively serialize containers
        if isinstance(obj, dict):
            return {k: serialize_for_hashing(v) for k, v in obj.items()}
        else:
            return [serialize_for_hashing(item) for item in obj]
    else:
        # For other types, return as-is (JSON serializable)
        return obj


def get_args_kwargs_str(*args, **kwargs) -> str:
    # Serialize args and kwargs, handling functions specially
    serialized_args = serialize_for_hashing(args)
    serialized_kwargs = serialize_for_hashing(kwargs)

    # Convert to JSON strings
    args_str = json.dumps(serialized_args, sort_keys=True)
    kwargs_str = json.dumps(serialized_kwargs, sort_keys=True)

    return args_str + kwargs_str


def get_dir_name(args_kwargs_str: str) -> str:
    hash_obj = hashlib.sha256(args_kwargs_str.encode())
    return f"{DATA_DIR}/{hash_obj.hexdigest()}"


def save_metadata(save_dir, metadata: dict):
    with (save_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def load_from_cache(save_dir) -> Dataset | None:
    if not save_dir.exists():
        raise FileNotFoundError(f"Cache directory {save_dir} not found")

    # Load metadata
    with (save_dir / "metadata.json").open("r") as f:
        metadata = json.load(f)

    metadata["last_accessed_at"] = datetime.datetime.now().isoformat()
    save_metadata(save_dir, metadata)

    with (save_dir / "data.pkl").open("rb") as f:
        return pickle.load(f)


def save_to_cache(source_fn, dataset, save_dir, args_kwargs_str: str | None = None):
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


def _dataset_from_source(
    source_fn: Callable,
    *args,
    shuffle: bool = False,
    seed: int | None = None,
    shuffle_choices: bool | int | None = None,
    limit: int | None = None,
    **kwargs,
) -> Dataset:
    args_kwargs_str = get_args_kwargs_str(*args, **kwargs)
    save_dir = Path(get_dir_name(args_kwargs_str))

    try:
        return load_from_cache(save_dir)
    except Exception:
        # I know this is general, but I want to catch all possible exceptions related to loading
        # the data or metadata, ok for now.
        pass

    dataset = source_fn(
        *args,
        **kwargs,
    )
    save_to_cache(source_fn, dataset, save_dir, args_kwargs_str)

    # shuffle if requested
    if shuffle:
        dataset.shuffle(seed=seed)

    shuffle_choices_if_requested(dataset, shuffle_choices)

    # limit if requested
    if limit:
        return dataset[0:limit]

    return dataset


def csv_dataset(
    *args,
    **kwargs,
) -> Dataset:
    return _dataset_from_source(
        csv_dataset_base,
        *args,
        **kwargs,
    )


def json_dataset(
    *args,
    **kwargs,
) -> Dataset:
    return _dataset_from_source(
        json_dataset_base,
        *args,
        **kwargs,
    )
