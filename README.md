# Label-Aware Calibration and Relation-Preserving in Visual Intention Understanding

Label-Aware Calibration and Relation-Preserving in Visual Intention Understanding

QingHongYa Shi, Mang Ye, Wenke Huang, Weijian Ruan, Bo Du. IEEE Transactions on Image Processing. 2024.

## Abstract

Visual intention understanding is a challenging task that explores the hidden intention behind the images of publishers in social media. Visual intention represents implicit semantics, whose ambiguous definition inevitably leads to label shifting and label blemish. The former indicates that the same image delivers intention discrepancies under different data augmentations, while the latter represents that the label of intention data is susceptible to errors or omissions during the annotation process. This paper proposes a novel method, called Label-aware Calibration and Relation-preserving (LabCR) to alleviate the above two problems from both intra-sample and inter-sample views. First, we disentangle the multiple intentions into a single intention for explicit distribution calibration in terms of the overall and the individual. Calibrating the class probability distributions in augmented instance pairs provides consistent inferred intention to address label shifting. Second, we utilize the intention similarity to establish correlations among samples, which offers additional supervision signals to form correlation alignments in instance pairs. This strategy alleviates the effect of label blemish. Extensive experiments have validated the superiority of the proposed method LabCR in visual intention understanding and pedestrian attribute recognition.

## Method Overview

![image](https://github.com/ShiQingHongYa/LabCR/blob/master/images/method.png)

We put the instance pairs $(v^1,v^2)$, which are augmented by data augmentation policies set $\mathcal{T}$ based on the original images, into the network $f_\theta$. The acquired feature pairs $(F^1,F^2)$ are fed into the IRP module for correlation matrix alignment. The DUDC module divides the logit outputs $(Z^1,Z^2)$ into independent logit of each target and aligns the class probability distribution from the overall and the individual.

## Environment

Install all required libraries:

```sh
pip install -r requirements.txt
```

## Quick Start

```sh
# training
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 train.py 
# evaluation
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 test.py
```

## Reference

If this work is useful to your research, please cite:

```sh
@article{shi2024label,
  title={Label-Aware Calibration and Relation-Preserving in Visual Intention Understanding},
  author={Shi, QingHongYa and Ye, Mang and Huang, Wenke and Ruan, Weijian and Du, Bo},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```
