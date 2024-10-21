# AutoDDH

## Graphic Abstract
![image](overview.png)

## Overview
AutoDDH is a deep learning framework designed to assist in the grading of Developmental Dysplasia of the Hip (DDH) using ultrasound images. This network leverages a dual attention mechanism and multi-task learning to enhance feature extraction and improve diagnostic accuracy.

## Dependencies
- Python 3.8+
- PyTorch 1.7+
- NumPy
- CUDA (for GPU acceleration)
- prefetch_generator
- os
- opencv-python
- glob
- logging
- re
- scikit-learn
- matplotlib
  
## Getting Started
### Data Preparation: Ensure that your dataset is organized in the following structure:
```
data/DDH/
├── train/
│   ├── img/
│   └── seg/
├── val/
│   ├── img/
│   └── seg/
└── test/
    ├── img/
    └── seg/

```
### Training: Run the training script using the provided command-line interface.
### Evaluation: Evaluate the model's performance on the test dataset.


## Usage
```bash
python train.py --trdir <path_to_training_data> --tedir <path_to_testing_data> --valdir <path_to_validation_data> --epochs <number_of_epochs> --lr <learning_rate> --batch-size <batch_size> --ncls <number_of_DDH_degrees> --nseg <number_of_key_structures>
```

## Authors
-Mengyao Liu
-Ruhan Liu

