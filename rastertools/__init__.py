from .version import __version__, __versiondate__, __license__  # noqa : F401
from rastertools.raster import raster_clip, raster_clip_weighted, raster_crop  # noqa : F401
from rastertools.shape import shape_subdivide  # noqa : F401


# module level doc-string
__doc__ = """
**rastertools** package provides features for extracting data from raster files using shapes/geometries as selectors.  
"""

# Use __all__ to let type checkers know what is part of the public API.
_all_ = ["raster_clip", "raster_clip_weighted", "raster_crop", "shape_subdivide"]
