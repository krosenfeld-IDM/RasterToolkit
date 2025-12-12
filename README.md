# RasterToolkit

RasterToolkit is a Python package for processing rasters with minimal dependencies. For example, with rastertoolkit you can extract populations corresponding to an administrative shapefile from a raster file.

## Setup

Install from github:

    python -m pip install .

## Getting Started

A typical `raster_clip` API usage scenario:
```
    from rastertoolkit import raster_clip

    # Clipping raster with shapes  
    pop_dict = raster_clip(raster_file, shape_file)  
```
See the complete code in the WorldPop example (examples/worldpop)

A typical `shape_subdivide` API usage scenario:
```
    from rastertoolkit import shape_subdivide

    # Create shape subdivision layer
    subdiv_stem = shape_subdivide(shape_stem=shape_file)
```
See the complete code in the Shape Subdivision example (examples/shape_subdivide)


## Running Tests

Install additional packages (like pytest)::
```
    python -m pip install .[test]
```
Run `pytest` command::
```
    # Run unit tests (recommended during development)
    python -m pytest -m unit -v

    # Run test for a specific module, for example
    python -m pytest tests/test_shape.py -v     # run shape unit tests
    python -m pytest tests/test_download.py -v  # run GDx download tests

    # All tests (before a commit or merging a PR)
    python -m pytest -v
```
