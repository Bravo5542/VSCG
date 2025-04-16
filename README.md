#  Leveraging the Video-level Semantic Consistency of Event for Audio-visual Event Localization.
## Yuanyuan Jiang (jyy55lesley@gmail.com), Jianqin Yin and Yonghao Dang 
PyTorch implementation of our TMM 2023 paper:  
[Leveraging the Video-level Semantic Consistency of Event for Audio-visual Event Localization](https://ieeexplore.ieee.org/abstract/document/10286391)

## Data Preparation
We highly appreciate [@YapengTian/AVE-ECCV2018](https://github.com/YapengTian/AVE-ECCV18) and [@Jinxing Zhou/PSP-CVPR2021](https://github.com/jasongief/PSP_CVPR_2021) for their great work and sharing.

The AVE dataset and the extracted audio and visual features can be downloaded from [here](https://github.com/YapengTian/AVE-ECCV18).

Other preprocessed files used in this repository can be downloaded from [here](https://drive.google.com/drive/folders/1q8GYBqfkyDDAnVMClrMTXR9YzH9UPcSM?usp=sharing).

All the required data are listed below, and these files should be placed into the ``data`` folder.
<pre><code>
audio_feature.h5  visual_feature.h5  audio_feature_noisy.h5 visual_feature_noisy.h5
right_label.h5  prob_label.h5  labels_noisy.h5  mil_labels.h5
train_order.h5  val_order.h5  test_order.h5
</code></pre>

## Fully supervised setting
- Train:
>  CUDA_VISIBLE_DEVICES=0 python fully_supervised_main.py --train
- Test:
>  CUDA_VISIBLE_DEVICES=0 python fully_supervised_main.py --trained_model_path ./model/VSCG_fully.pt

## Weakly supervised setting
- Train:
> CUDA_VISIBLE_DEVICES=0 python weakly_supervised_main.py --train
- Test:
> CUDA_VISIBLE_DEVICES=0 python weakly_supervised_main.py --trained_model_path ./model/VSCG_weakly.pt

**Note:** The pre-trained models can be downloaded [here](https://drive.google.com/drive/folders/1_wcW1T7HeLSkEOYTTRj1PnHXWReeaynT?usp=share_link) and they should be placed into the ``model`` folder.

## Citation
If our paper is useful for your research, please consider citing it:
```ruby
@ARTICLE{vscg2023,
  author={Jiang, Yuanyuan and Yin, Jianqin and Dang, Yonghao},
  journal={IEEE Transactions on Multimedia}, 
  title={Leveraging the Video-level Semantic Consistency of Event for Audio-visual Event Localization}, 
  year={2023},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TMM.2023.3324498}}


```
