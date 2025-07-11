from pathlib import Path

CACHE_DATA = True  # if false then no caching is performed

# where to store cached data, this is base level of the repo. Don't forget to add this to .gitignore
DATA_DIR = Path(__file__).parent.parent.parent.parent / "cached_data"
