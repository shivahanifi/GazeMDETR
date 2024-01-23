# GazeMDETR 

This repository contains the required files for combining gaze information with original MDETR.

## Original MDETR repository

**MDETR**: Modulated Detection for End-to-End Multi-Modal Understanding

[Website](https://ashkamath.github.io/mdetr_page/) • [Colab](https://colab.research.google.com/drive/11xz5IhwqAqHj9-XAIP17yVIuJsLqeYYJ?usp=sharing) • [Paper](https://arxiv.org/abs/2104.12763)

## GazeMDETR Usage
- First time using the repo:

  Make a new conda env and activate it: 
  ```
  conda create -n mdetr_env python=3.8
  conda activate mdetr_env
  ```

  Install the the packages in the requirements.txt:
  ```
  pip install -r requirements.txt
  ```

- Repetative usage:

  Activate the environment in VS code (Ctrl+shif+p) and select `mdetr_env2`(or any name you have selected).

## Test Data

The data collected to test the MDETR and GazeMDETR are accessible through the following links:
- [raw_hm_dump](https://drive.google.com/drive/folders/1D0NyE2SpGJ9DiHIgd8LhoQUhkywDq-Nu?usp=sharing)
- [MDETR_test_data](https://drive.google.com/drive/folders/1yrsScizASYnUpczeBKNaPRvDlvtHZALA?usp=sharing)
- []()

## Combining gaze information with the MDETR data

1. As a first step, I make the heatmap, which is the output of the VTD, available in the GazeMDETR demo code, and do the initial tests on how it would be possible to combine it with the features that are output of the backbone in the MDETR. To that end, the heatmap is resized and converted to tensor and then downsampled to the size of the features, such that they can be multiplied. The visualized output is presented below:

    <img src="img/norm_map_before_after_downsample.png">