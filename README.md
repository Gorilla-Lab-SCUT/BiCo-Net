# BiCo-Net
Code for "BiCo-Net: Regress Globally, Match Locally for Robust 6D Pose Estimation" [[Arxiv](https://arxiv.org/abs/2205.03536)][[Paper](https://www.ijcai.org/proceedings/2022/210)]

![image](https://github.com/Gorilla-Lab-SCUT/BiCo-Net/blob/main/doc/network.png)

## Requirements
This code has been tested with
- Open3D==0.9.0.0
- Python==3.7.12
- OpenCV==4.1
- Pytorch==1.6.0
- CUDA==10.1

## Downloads
- YCB-Video dataset [[link](https://rse-lab.cs.washington.edu/projects/posecnn)]
- Preprocessed LineMOD dataset provided by DenseFusion [[link](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7)]
- Pretrained models of BiCo-Net [[link](https://drive.google.com/drive/folders/18n506Vww8NipgWYxkVH0Cl2YIeufRZXz?usp=sharing)]
- Pred mask of PVN3D on YCB-Video dataset [[link](https://drive.google.com/file/d/1ftLn9itGQtjx5QM7SfOousIL44olIcm9/view?usp=sharing)]
- Pred mask of HybridPose on LineMOD Occlusion dataset [[link](https://drive.google.com/file/d/1Jwp-J6opAAvtbMV1ewzhpBLoSjmZoMVJ/view?usp=sharing)]

## Training
Command for training BiCo-Net:
```
python train.py --dataset ycbv --dataset_root path_to_ycbv_dataset
python train.py --dataset linemod --dataset_root path_to_lm_dataset
```

For lmo dataset, download the [VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and run:

```
python train.py --dataset lmo --dataset_root path_to_lm_dataset --bg_img path_to_voc2012_dataset
```

## Evaluation
Evaluate the results of BiCo-Net reported in the paper:
```
python eval_ycbv.py --dataset_root path_to_ycbv_dataset --pred_mask path_to_pvn3d_pred_mask
python eval_lm.py --dataset_root path_to_lm_dataset
python eval_lmo.py --dataset_root path_to_lm_dataset --pred_mask path_to_hybridpose_pred_mask
```

## Acknowledgements
Our implementation leverages the code from [DenseFusion](https://github.com/j96w/DenseFusion).

## License
Our code is released under MIT License (see LICENSE file for details).

## Citation
If you find our work useful in your research, please consider citing:

     @inproceedings{Xu2022BiCoNetRG,
       title={BiCo-Net: Regress Globally, Match Locally for Robust 6D Pose Estimation},
       author={Zelin Xu and Yichen Zhang and Ke Chen and Kui Jia},
       booktitle={IJCAI},
       year={2022}
     }
