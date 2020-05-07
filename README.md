# Aggregating Attentional Dilated Features for Salient Object Detection
by Lei Zhu^, Jiaxing Chen^, Xiaowei Hu, Chi-Wing Fu, Xuemiao Xu, Jing Qin, and Pheng-Ann Heng (^ joint 1st authors)[[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8836095)]

This implementation is written by Jiaxing Chen at the South China University of Technology.

## Citation

@article{zhu2019aggregating,   
&nbsp;&nbsp;&nbsp;&nbsp;  title={Aggregating Attentional Dilated Features for Salient Object Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  author={Zhu, Lei and Chen, Jiaxing and Hu, Xiaowei and Fu, Chi-Wing and Xu, Xuemiao and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE Transactions on Circuits and Systems for Video Technology},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2019},    
&nbsp;&nbsp;&nbsp;&nbsp;  publisher={IEEE}    
}

## Saliency Map

The results of salient object detection on seven datasets (ECSSD, HKU-IS, PASCAL-S, SOD, DUT-OMRON, DUTS-TE, SOC) can be found at [Google Drive](https://drive.google.com/open?id=1tv72yWNH0ANHoSU4qMOwD7g5r53wSZEe).

## Trained Model

You can download the trained model which is reported in our paper at  [Google Drive](https://drive.google.com/file/d/1AWFG6x2lLNTUttBIxzrCFH277wkRuFX-/view?usp=sharing).

## Requirement

- Python 2.7
- PyTorch 0.4.0
- torchvision
- numpy
- Cython
- pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Training

1. Set the path of pretrained DenseNet model in densenet/config.py
2. Set the path of DUTS dataset in config.py
3. Run by `python train.py`

*Hyper-parameters* of training were gathered at the beginning of *train.py* and you can conveniently change them as you need.

## Testing

1. Set the path of six benchmark datasets in config.py
2. Put the trained model in ckpt/AADFNet
3. Run by `python infer.py`

*Settings* of testing were gathered at the beginning of *infer.py* and you can conveniently change them as you need.

## Dataset links

- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://sites.google.com/site/ligb86/hkuis), [PASCAL-S](http://cbi.gatech.edu/salobj/), [SOD](http://elderlab.yorku.ca/SOD/), [DUT-OMRON](http://ice.dlut.edu.cn/lu/DUT-OMRON/Homepage.htm), [DUTS](http://saliencydetection.net/duts/), [SOC](http://dpfan.net/SOCBenchmark/), : the seven benchmark datasets
