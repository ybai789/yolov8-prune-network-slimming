### Introduction

This repository provides a pruning method for YOLOv8, leveraging the network slimming approach. 

It adapts the codebase of YOLOv8 version 8.1.33 to implement this method. 

For an in-depth understanding of the underlying principles, it's recommended to consult the research paper titled "Learning Efficient Convolutional Networks through Network Slimming," available at https://arxiv.org/abs/1708.06519. 

This method aims at enhancing the efficiency and performance of YOLOv8 by slimming down the network, aligning with the strategies outlined in the cited paper.

### Installation

Clone the repository and rename the `yolov8-prune-network-slimming` directory to `ultralytics`.

```
cd ultralytics

pip install -e .
```

### Steps

Within the  `ultralytics` directory, execute the scripts provided below. 

The parameters within these Python scripts can be adjusted according to your requirements.

1. Basic training
    ```shell
    python train.py
    ```
2. Sparse training
    ```shell
    python train_sparsity.py
    ```

3. Pruning
    ```shell
    python prune.py
    ```

4. Fine-tuning
    ```shell
    python finetune.py
    ```
### Experiments
  Result of YOLOv8s pruning on PASCAL VOC Dataset 

| Description                 | mAP50 | mAP50-95 | Params/FLOPs |
| --------------------------- | ----- | -------- | ------------ |
| Original                    | 0.864 | 0.674    | 11.1M/28.5G  |
| Sparse Training             | 0.856 | 0.658    | 11.1M/28.5G  |
| 30% Pruning and Fine-tuning | 0.858 | 0.665    | 6.2M/20.4G   |