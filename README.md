# Awesome-Knowledge-Fusion
[![Awesome](https://awesome.re/badge.svg)]()
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt=""/>

<!-- > [!TIP] -->
<font color="red">If you have any questions about the library, please feel free to contact us.
Email: duguodong7@gmail.com</font>

---

A comprehensive list of papers about **'[Knowledge Fusion: The Integration of Model Capabilities.]'**.

## Abstract
> As the comprehensive capabilities of foundational large models rapidly improve, similar general abilities have emerged across different models, making capability transfer and fusion between them more feasible. Knowledge fusion aims to integrate existing LLMs of diverse architectures and capabilities into a more powerful model through efficient methods such as knowledge distillation, model merging, mixture of experts, and PEFT, thereby reducing the need for costly LLM development and adaptation. We provide a comprehensive overview of model merging methods and theories, covering their applications across various fields and scenarios, including LLMs, MLLMs, image generation, model compression, continual learning, and more. Finally, we highlight the challenges of knowledge fusion and explore future research directions.

<center>
<img src="./imgs/knowledge_fusion.png" alt="knowledge_fusion" width="800"/>
</center>

******

## Framework
- [Awesome-Knowledge-Fuse](#awesome-model-merging-methods-theories-applications)
  * [1. Connectivity and Alignment](#1-connectivity-and-alignment)
    + [1.1 Model Connectivity](#11-model-connectivity)
    + [1.2 Weight Alignment](#12-weight-alignment)
  * [2. Parameter Merging](#2-parameter-merging)
    + [2.1 Merging Methods](#21-merging-methods)
      - [Gradient based](#gradient-based)
      - [Task Vector based](#task-vector-based)
    + [2.2 During or After Training](#22-during-or-after-training)
      - [During Training](#during-training)
      - [After Training](#after-training)
    + [2.3 For LLMs and MLLMs](#23-for-llms-and-mllms)
      - [For LLMs](#for-llms)
      - [For MLLMs](#for-mllms)
  * [3. Model Ensemble](#3-model-ensemble)
    + [3.1 Ensemble Methods](#31-ensemble-methods)
      - [Weighted Averaging](#weighted-averaging)
      - [Routing](#routing)
      - [Voting](#Voting)
    + [3.2 Ensemble Object](#32-ensemble-object)
      - [Entire Model](#entire-model)
      - [Adapter](#Adapter)
  * [4. Decouple and Reuse](#4-decouple-and-reuse)
    + [4.1 Reprogramming](#41-reprogramming)
    + [4.2 Mask](#42-mask)
  * [5. Distillation](#5-distillation)
    + [5.1 Transformer](#51-transformer)
    + [5.2 CNN](#52-cnn)
    + [5.3 GNN](#53-gnn)
  * [6. Model Reassemble](#6-model-reassemble)
    + [6.1 Model Stitch](#61-model-stitch)
    + [6.2 Model Evolution](#62-model-evolution)
  * [7. Others](#7-others)
    + [7.1 External Data Retrieval](#71-external-data-retrieval)
    + [7.2 Other Surveys](#72-other-survyes)
----------

## 1. Connectivity and Alignment
#### 1.1 Model Connectivity
| **Paper Title** | **Key Word** | Code | **Pub & Date** |
| --------------- | :----: | :----: |
| [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf) | LMC | | ICML2020 |
| [On Convexity And Linear Mode Connectivity In Neural Networks](https://opt-ml.org/papers/2022/paper90.pdf) | workshop | | OPT2022 |
| [Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling](https://proceedings.mlr.press/v139/benton21a/benton21a.pdf) |  | | ICML2021 |
| [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://papers.nips.cc/paper_files/paper/2018/file/be3087e74e9100d4bc4c6268cdbe8456-Paper.pdf)|   |  | NeurIPS2018 |
| [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/pdf/2209.04836) | 2023 | ICLR |
| [Re-basin via implicit Sinkhorn differentiation](https://openaccess.thecvf.com/content/CVPR2023/papers/Pena_Re-Basin_via_Implicit_Sinkhorn_Differentiation_CVPR_2023_paper.pdf) | 2023 | CVPR |
| [Plateau in Monotonic Linear Interpolation--A "Biased" View of Loss Landscape for Deep Networks](https://arxiv.org/pdf/2210.01019)| 2023 | ICLR |
| [Linear Mode Connectivity of Deep Neural Networks via Permutation Invariance and Renormalization](https://openreview.net/pdf?id=gU5sJ6ZggcX)| 2023 | ICLR |
| [Rethink Model Re-Basin and the Linear Mode Connectivity](https://arxiv.org/pdf/2402.05966) | 2024 | Arxiv |
| [Going beyond linear mode connectivity: The layerwise linear feature connectivity](https://papers.nips.cc/paper_files/paper/2023/file/bf3ee5a5422b0e2a88b0c9c6ed3b6144-Paper-Conference.pdf) |  2023 |NeurIPS |
| [Layerwise linear mode connectivity](https://openreview.net/pdf?id=LfmZh91tDI) | 2024 | ICLR |
| [Proving linear mode connectivity of neural networks via optimal transport](https://arxiv.org/pdf/2310.19103) | 2024 | AISTATS |
| [The role of permutation invariance in linear mode connectivity of neural networks](https://openreview.net/pdf?id=dNigytemkL) | 2022 | ICLR |
| [What can linear interpolation of neural network loss landscapes tell us?](https://arxiv.org/pdf/2106.16004) |2022 | ICML |
| [Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling](https://proceedings.mlr.press/v139/benton21a/benton21a.pdf) | 2021 | ICML |
| [Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes](https://proceedings.mlr.press/v139/lucas21a/lucas21a.pdf) | 2021 | ICML |
| [Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances](https://proceedings.mlr.press/v139/simsek21a/simsek21a.pdf)| 2021 | ICML |


#### 1.2 Weight Alignment
| **Paper Title** | **Year** | **Conference/Journal** |
| [Equivariant Deep Weight Space Alignment](https://openreview.net/pdf/6d437eeb362255b4b2d75a5c6847880fb4a00e3c.pdf) | 2024 | ICML  |
| [Harmony in diversity: Merging neural networks with canonical correlation analysis](https://openreview.net/pdf?id=XTr8vwAr2D) | 2024 | ICML |
| [Transformer fusion with optimal transport](https://arxiv.org/pdf/2310.05719) | 2024 | ICLR  |
| [Merging Models From Different Tasks Without Training](https://arxiv.org/pdf/2310.05719) | 2024 | ICLR  |
| [Training-Free Pretrained Model Merging](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Training-Free_Pretrained_Model_Merging_CVPR_2024_paper.pdf) | 2024 |CVPR  |
| [Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering](https://arxiv.org/pdf/2409.16167)| 2024 | Arxiv |
| [C2M3: Cycle-Consistent Multi Model Merging](https://arxiv.org/pdf/2405.17897) | 2024 | Arxiv |

| [REPAIR: REnormalizing Permuted Activations for Interpolation Repair](https://openreview.net/pdf?id=gU5sJ6ZggcX) |2023  | ICLR |

| [Optimizing mode connectivity via neuron alignment](https://arxiv.org/pdf/2009.02439) | 2020 | NeurIPS |
| [Model fusion via optimal transport](https://proceedings.neurips.cc/paper/2020/file/fb2697869f56484404c8ceee2985b01d-Paper.pdf) | 2020  | NeurIPS |
| [Uniform convergence may be unable to explain generalization in deep learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/05e97c207235d63ceb1db43c60db7bbb-Paper.pdf) |  2019 | NeurIPS |
| [Explaining landscape connectivity of low-cost solutions for multilayer nets](https://proceedings.neurips.cc/paper_files/paper/2019/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf)|  2019 | NeurIPS |
| [Essentially no barriers in neural network energy landscape](https://proceedings.mlr.press/v80/draxler18a/draxler18a.pdf) | 2018 | ICML  |
| [Weight Scope Alignment: A Frustratingly Easy Method for Model Merging](https://arxiv.org/pdf/2408.12237) | 2024 | Arxiv |

## 2. Parameter Merging
### 2.1 Merging Methods
#### Gradient based
| **Paper Title** | **Key Word** | Code | **Pub & Date** |
| --------------- | :----: | :----: | :----: |


#### Task Vector based
| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: | :----: |


### 2.2 During or After Training
#### During Training

#### After Training

### 2.3 For LLMs and MLLMs
#### For LLMs
#### For MLLMs

## 3. Model Ensemble

### 3.1 Ensemble Methods
#### Weighted Averaging
#### Routing
#### Voting

### 3.2 Ensemble Object
#### Entire Model
#### Adapter

## 4. Decouple and Reuse
### 4.1 Reprogramming
### 4.2 Mask

## 5. Distillation
### 5.1 Transformer
### 5.2 CNN
### 5.3 GNN

## 6. Model Reassemble
### 6.1 Model Stitch
### 6.2 Model Evolution

## 7. Others
### 7.1 External Data Retrieval
### 7.2 Other Surveys

----------

**Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=duguodong7/Awesome-Knowledge-Fusion&type=Date)](https://star-history.com/#duguodong7/Awesome-Knowledge-Fusion&Date)

----------


## Contact
<!-- **Contact** -->

We invite all researchers to contribute to this repository, **'Knowledge Fusion: The Integration of Model Capabilities'**.
If you have any questions about the library, please feel free to contact us.

Email: duguodong7@gmail.com
