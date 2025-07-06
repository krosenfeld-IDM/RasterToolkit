# RasterToolkit

RasterToolkit is a Python package for processing rasters with minimal dependencies. For example, with rastertoolkit you can extract populations corresponding to an administrative shapefile from a raster file.

## Setup

Install from github:

    pip install git+https://github.com/InstituteforDiseaseModeling/RasterToolkit.git

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

## Developer Setup 

1. Clone or download this GitHub repo and navigate to the root directory.
```
    git clone git@github.com:InstituteforDiseaseModeling/RasterToolkit.git
    cd RasterToolkit
```
2. Create a Python virtual environment (here useing `uv`; see https://astral.sh/uv/):
```
    uv venv --python 3.10
    source .venv/bin/activate
```
3. Install this package in editable mode (this also installs all the requirements).::
```
    uv pip install -e .   
```

## Running Tests

Install additional packages (like pytest)::
```
    uv pip install -r requirements-test.txt
```
Run `pytest` command::
```
    # Run unit tests (recommended during development)
    pytest -m unit -v

    # Run test for a specific module, for example
    pytest tests/test_shape.py -v     # run shape unit tests
    pytest tests/test_download.py -v  # run GDx download tests

    # All tests (before a commit or merging a PR)
    pytest -v
```
