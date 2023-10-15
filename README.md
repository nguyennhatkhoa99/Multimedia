# Multi Media

Vision Transformer Hashing (VTS) utilizes the Vision Transformer to generate the hash code for image retrieval.
It is tested under retrieval frameworks such as DSH.

## How to Run

This code uses the Vision Transformer (ViT) code and pretrained model (https://github.com/jeonsworld/ViT-pytorch) and DeepHash framework (https://github.com/swuxyj/DeepHash-pytorch).

There are 3 files as database, train and test in each dataset for data path and binary hashing code index file.

For running CIFAR-10 dataset image retrieve with file path as query image agrument. Do as following:
```
python DSHcls.py --query-image="/path/to/the/image/in/test/dataset"
```
