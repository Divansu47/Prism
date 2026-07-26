# Segmentation training data layout

Use this folder as a small starter dataset for the segmentation pipeline.

## Folder structure
- images/: RGB input images
- masks/: grayscale segmentation masks
- labels_template.csv: starter label file for local training
- schema.json: machine-readable schema for the CSV columns

## Mask convention
- 0 = background
- 1 = rice_staple
- 2 = curry_gravy
- 3 = vegetable
- 4 = protein

## Label CSV schema
The training CSV should contain these columns:
- image_path: relative path to the image file
- mask_path: relative path to the mask file
- rice_staple: placeholder column (use 1 for a sample that contains that class, 0 otherwise)
- curry_gravy: placeholder column
- vegetable: placeholder column
- protein: placeholder column

You can replace the placeholder values later with richer annotations if you collect more data.
