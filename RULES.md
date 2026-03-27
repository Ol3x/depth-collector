The project needs to use one micromamba environment that will be named depth-collector.

As much as possible, using external libraries should be avoided.

Most of the data processing should be done using numpy only.

A lot of things will be configurable, but ALL the config for one pipeline project should be contained in one unique file (for now we will use only a default config file).

The code should use abstract classes as possible to define the pipeline and preprocessing logic.

There will be an abstract DatasetPipeline class that we'll use for all datasets (MegaDepth, Hypersim, etc).

For each dataset, the concrete class should implement methods to

- download the data using hugging face hub
- extract the data and remove archives
- process the data using numpy
- store the data as pytorch tensor contained in .pt shards, contained in .tar files, according to the webdataset format

EXTREMELY IMPORTANT : for all datasets, the data should be in DISTANCE TO THE CAMERA format, not depth. To achieve that, depending on the camera model of the dataset, different operations may be done to compute the distance to the camera.

In general, consider that two different "samples" can have different height and width.

One sample will always consist of :

- a (H,W,3) image in RGB order
- a (H,W,1) distance grid
- a (H,W,3) ray directions tensor

In general, datasets folders should contain a metadata.json file indicating the number of shards as well as suggested training and validation files (based on a train_val_split config number between 0 and 1) and number of files per shard.

In general, shards should hold 1GB of data.

Such that, for each pixel, the 3D point corresponds to the distance along the ray, obtained via the product pts = dist * ray_dir.

Each dataset will have a specific folder that will contain : 
- raw
- processed
    - files
        - .tar shards
    - metadata.json