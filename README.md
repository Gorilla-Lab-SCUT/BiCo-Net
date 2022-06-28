# BiCo-Net
Code for "BiCo-Net: Regress Globally, Match Locally for Robust 6D Pose Estimation" [[Arxiv](https://arxiv.org/abs/2205.03536)]

![image](https://github.com/Gorilla-Lab-SCUT/BiCo-Net/blob/main/doc/network.pdf)

## Requirements
This code has been tested with
- python 3.7.12
- pytorch 1.6.0
- CUDA 10.1

## Downloads
- YCB-Video dataset [[link](https://rse-lab.cs.washington.edu/projects/posecnn)]
- Preprocessed LineMOD dataset provided by DenseFusion [[link](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7)]
- Pretrained models of BiCo-Net [[link](https://drive.google.com/drive/folders/18n506Vww8NipgWYxkVH0Cl2YIeufRZXz?usp=sharing)]
- Pred mask of PVN3D on YCB-Video dataset [[link](https://drive.google.com/file/d/1ftLn9itGQtjx5QM7SfOousIL44olIcm9/view?usp=sharing)]

## Training
Command for training BiCo-Net:
```
python train.py --dataset ycbv --dataset_root path_to_ycbv_dataset
python train.py --dataset linemod --dataset_root path_to_lm_dataset
```

## Evaluation
Evaluate the results of BiCo-Net reported in the paper:
```
python eval_ycbv.py --dataset_root path_to_ycbv_dataset --pred_mask path_to_pvn3d_pred_mask
python eval_lm.py --dataset_root path_to_lm_dataset
```

## Acknowledgements
Our implementation leverages the code from [DenseFusion](https://github.com/j96w/DenseFusion).

## License
Our code is released under MIT License (see LICENSE file for details).

## Citation
If you find our work useful in your research, please consider citing:

     @article{xu2022bico,
         title     = {BiCo-Net: Regress Globally, Match Locally for Robust 6D Pose Estimation},
         author    = {Xu, Zelin and Zhang, Yichen and Chen, Ke and Jia, Kui},
         journal   = {arXiv preprint arXiv:2205.03536},
         year      = {2022}
     }
