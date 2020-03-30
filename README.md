# 3D MRI Segmentation
Implementation of deep residual 3D U-Net and an MRI segmentation pipeline in PyTorch. 

Source code for *[Automatic Post-Stroke Lesion Segmentation on MR Images using 3D Residual Convolutional Neural Network](https://arxiv.org/abs/1911.11209)*

# Dependencies
- Python 3.6
- PyTorch 1.3
- NiBabel
- scikit-image
- scikit-learn


# Usage
## Training a 3D segmentation model
- `python train.py --data_dir <path_to_dataset> --save_dir <path_to_snapshots>`

## Testing a trained model
- `python test.py --data_dir <path_to_dataset> --model_path <path_to_a_snapshot>`


# Citations
PrepSlide is an open-source library and is licensed under the GNU General Public License (v3). For questions contact Saeed Hassanpour at Saeed.Hassanpour@dartmouth.edu.
