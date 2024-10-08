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
      - [For Multimodal Language Models](#for-multimodal-language-models)
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
    + [6.1 Model Evolution](#61-model-evolution)
  * [7. Others](#7-others)
    + [7.1 External Data Retrieval](#71-external-data-retrieval)
    + [7.2 Other Surveys](#72-other-survyes)
----------

## 1. Connectivity and Alignment
#### 1.1 Model Connectivity
| **Paper Title** | **Code** | **Pub & Date** |
| --------------- | :----: | :----: |
| [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf) | LMC | |
| [On Convexity And Linear Mode Connectivity In Neural Networks](https://opt-ml.org/papers/2022/paper90.pdf) | workshop | |
| [Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling](https://proceedings.mlr.press/v139/benton21a/benton21a.pdf) |  | |
| [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://papers.nips.cc/paper_files/paper/2018/file/be3087e74e9100d4bc4c6268cdbe8456-Paper.pdf)|   |  |
| [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/pdf/2209.04836) | [git re-basin](https://github.com/samuela/git-re-basin) | **ICLR** 2023 |
| [Re-basin via implicit Sinkhorn differentiation](https://openaccess.thecvf.com/content/CVPR2023/papers/Pena_Re-Basin_via_Implicit_Sinkhorn_Differentiation_CVPR_2023_paper.pdf) |  | **CVPR** 2023 |
| [Plateau in Monotonic Linear Interpolation--A "Biased" View of Loss Landscape for Deep Networks](https://arxiv.org/pdf/2210.01019)|  | **ICLR** 2023 |
| [Linear Mode Connectivity of Deep Neural Networks via Permutation Invariance and Renormalization](https://openreview.net/pdf?id=gU5sJ6ZggcX)|  | **ICLR** 2023 |
| [Rethink Model Re-Basin and the Linear Mode Connectivity](https://arxiv.org/pdf/2402.05966) |  | **Arxiv** 2024 |
| [Going beyond linear mode connectivity: The layerwise linear feature connectivity](https://papers.nips.cc/paper_files/paper/2023/file/bf3ee5a5422b0e2a88b0c9c6ed3b6144-Paper-Conference.pdf) |   |**NeurIPS** 2023 |
| [Layerwise linear mode connectivity](https://openreview.net/pdf?id=LfmZh91tDI) |  | **ICLR** 2024 |
| [Proving linear mode connectivity of neural networks via optimal transport](https://arxiv.org/pdf/2310.19103) |  | **AISTATS** 2024 |
| [The role of permutation invariance in linear mode connectivity of neural networks](https://openreview.net/pdf?id=dNigytemkL) |  | **ICLR** 2022 |
| [What can linear interpolation of neural network loss landscapes tell us?](https://arxiv.org/pdf/2106.16004) | | **ICML** 2022 |
| [Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling](https://proceedings.mlr.press/v139/benton21a/benton21a.pdf) |  | **ICML** 2021 |
| [Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes](https://proceedings.mlr.press/v139/lucas21a/lucas21a.pdf) |  | **ICML** 2021 |
| [Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances](https://proceedings.mlr.press/v139/simsek21a/simsek21a.pdf)|  | **ICML** 2021 |


#### 1.2 Weight Alignment
| **Paper Title**                                              | **Code** | **Pub & Date      ** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Equivariant Deep Weight Space Alignment](https://openreview.net/pdf/6d437eeb362255b4b2d75a5c6847880fb4a00e3c.pdf) |  | **ICML** 2024 |
| [Harmony in diversity: Merging neural networks with canonical correlation analysis](https://openreview.net/pdf?id=XTr8vwAr2D) | [CCA Merge](https://github.com/shoroi/align-n-merge) | **ICML** 2024 |
| [Transformer fusion with optimal transport](https://arxiv.org/pdf/2310.05719) |  | **ICLR** 2024 |
| [ZipIt! Merging Models From Different Tasks Without Training](https://arxiv.org/pdf/2310.05719) |  | **ICLR** 2024 |
| [Training-Free Pretrained Model Merging](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Training-Free_Pretrained_Model_Merging_CVPR_2024_paper.pdf) |  |**CVPR** 2024  |
| [Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering](https://arxiv.org/pdf/2409.16167)|  | **Arxiv** 24.09 |
| [C2M3: Cycle-Consistent Multi Model Merging](https://arxiv.org/pdf/2405.17897) |  | **ArXiv** 24.05 |
| [REPAIR: REnormalizing Permuted Activations for Interpolation Repair](https://openreview.net/pdf?id=gU5sJ6ZggcX) |[REPAIR](https://github.com/KellerJordan/REPAIR)  | **ICLR** 2023 |
| [Optimizing mode connectivity via neuron alignment](https://arxiv.org/pdf/2009.02439) |  | **NeurIPS** 2020 |
| [Model fusion via optimal transport](https://proceedings.neurips.cc/paper/2020/file/fb2697869f56484404c8ceee2985b01d-Paper.pdf) | [otfusion](https://github.com/sidak/otfusion) | **NeurIPS** 2020 |
| [Uniform convergence may be unable to explain generalization in deep learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/05e97c207235d63ceb1db43c60db7bbb-Paper.pdf) |   | **NeurIPS** 2019 |
| [Explaining landscape connectivity of low-cost solutions for multilayer nets](https://proceedings.neurips.cc/paper_files/paper/2019/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf)|   | **NeurIPS** 2019 |
| [Essentially no barriers in neural network energy landscape](https://proceedings.mlr.press/v80/draxler18a/draxler18a.pdf) |  | **ICML** 2018 |
| [Weight Scope Alignment: A Frustratingly Easy Method for Model Merging](https://arxiv.org/pdf/2408.12237) |  | **ArXiv** 24.08 |

## 2. Parameter Merging
### 2.1 Merging Methods
#### Gradient based

| **Paper Title** | **Code** | **Pub & Date** |
| --------------- | :------: | :------------: |


#### Task Vector based

### 2.2 During or After Training
#### During Training

#### After Training

### 2.3 For LLMs and MLLMs
#### For LLMs
#### For Multimodal Language Models

## 3. Model Ensemble

### 3.1 Ensemble Methods
#### Weighted Averaging
#### Routing
#### Voting

### 3.2 Ensemble Object
#### Entire Model
| **Paper Title**                                              | **Code** | **Pub & Date** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning](https://proceedings.mlr.press/v162/liu22k/liu22k.pdf) | [GAME](https://github.com/Thinklab-SJTU/GAMF) | **ICML **2022 |
| [Merging](https://arxiv.org/abs/2402.00433) |          | **ICML **2024 |
#### Adapter

| **Paper Title**                                              |                           **Code**                           | **Pub & Date** |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :------------: |
| [Mixture-of-Domain-Adapters: Decoupling and Injecting Domain Knowledge to Pre-trained Language Models' Memories ](https://arxiv.org/abs/2306.05406) | [code](https://github.com/xu1868/Mixture-of-Domain-Adapters) |  **ACL **2023  |
| [Merging Multi-Task Models via Weight-Ensembling Mixture of Experts](https://arxiv.org/abs/2402.00433) |  [WEMOE](https://github.com/tanganke/weight-ensembling_moe)  | **ICML **2024  |

## 4. Decouple and Reuse
### 4.1 Reprogramming

| **Paper Title**                                              |                           **Code**                           |  **Pub & Date**  |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :--------------: |
| [Model Reprogramming: Resource-Efficient Cross-Domain Machine Learning](https://ojs.aaai.org/index.php/AAAI/article/view/30267) |                                                              |  **AAAI **2024   |
| [Towards Efficient Task-Driven Model Reprogramming with Foundation Models](https://arxiv.org/pdf/2304.02263) |                                                              | **ArXiv** 23.06  |
| [From English to More Languages: Parameter-Efficient Model Reprogramming for Cross-Lingual Speech Recognition](https://ieeexplore.ieee.org/abstract/document/10094903) |                                                              | **ICASSP** 2023  |
| [Deep Graph Reprogramming](https://openaccess.thecvf.com/content/CVPR2023/papers/Jing_Deep_Graph_Reprogramming_CVPR_2023_paper.pdf) |             [ycjing](https://github.com/ycjing)              |  **CVPR **2023   |
| [Fairness Reprogramming](https://proceedings.neurips.cc/paper_files/paper/2022/file/de08b3ee7c0043a76ee4a44fe68e90bc-Paper-Conference.pdf) | [USBC-NLP](https://github.com/UCSB-NLP-Chang/Fairness-Reprogramming) | **NeurIPS **2022 |
| [Voice2Series: Reprogramming Acoustic Models for Time Series Classification](https://proceedings.mlr.press/v139/yang21j/yang21j.pdf) | [V2S](https://github.com/huckiyang/Voice2Series-Reprogramming) |  **ICML** 2021   |

### 4.2 Mask

| **Paper Title**                                              |                          **Code**                           |      **Pub & Date**       |
| ------------------------------------------------------------ | :---------------------------------------------------------: | :-----------------------: |
| [EMR-Merging: Tuning-Free High-Performance Model Merging](https://arxiv.org/pdf/2405.17461) | [EMR_Merging](https://github.com/harveyhuang18/EMR_Merging) | **NeurIPS **2024 spolight |
| [Model Composition for Multimodal Large Language Models](https://arxiv.org/pdf/2402.12750) |     [THUNLP](https://github.com/THUNLP-MT/ModelCompose)     |       **ACL** 2024        |
| [Localizing Task Information for Improved Model Merging and Compression](https://openreview.net/attachment?id=DWT9uiGjxT&name=pdf) |     [tall_masks](https://github.com/nik-dim/tall_masks)     |       **ICML **2024       |
| [Adapting a Single Network to Multiple Tasks by Learning to Mask Weights](https://openaccess.thecvf.com/content_ECCV_2018/papers/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.pdf) |                          Piggyback                          |       **ECCV **2018       |

## 5. Distillation
### 5.1 Transformer

| **Paper Title**                                              | **Code** | **Pub & Date** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Knowledge Fusion of Chat LLMs: A Preliminary Technical Report](https://arxiv.org/pdf/2402.16107) | [FuseChat](https://github.com/18907305772/FuseAI) | **ArXiv** 24.02 |
| [Sam-clip: Merging vision foundation models towards semantic and spatial understanding](https://openaccess.thecvf.com/content/CVPR2024W/eLVM/html/Wang_SAM-CLIP_Merging_Vision_Foundation_Models_Towards_Semantic_and_Spatial_Understanding_CVPRW_2024_paper.html) |  | **CVPR** 2024 |
| [Knowledge fusion of large language models](https://openreview.net/pdf?id=jiDsk12qcz) | [FuseAI](https://github.com/18907305772/FuseAI) | **ICLR** 2024 |
| [Seeking Neural Nuggets: Knowledge Transfer In Large Language Models From A Parametric Perspective](https://arxiv.org/pdf/2310.11451) | [ParaKnowTransfer](https://maszhongming.github.io/ParaKnowTransfer/) | **ICLR** 2024 |
| [One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation](https://arxiv.org/pdf/2310.19444) | [OFAKD](https://github.com/Hao840/OFAKD) | **NeurIPS** 2023 |
| [Knowledge Amalgamation for Object Detection With Transformers](https://ieeexplore.ieee.org/document/10091778) |  | **TIP** 2023 |

### 5.2 CNN

| **Paper Title**                                              | **Code** | **Pub   &           Date** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Factorizing Knowledge in Neural Networks](https://arxiv.org/pdf/2207.03337.pdf) | [KnowledgeFactor](https://github.com/Adamdad/KnowledgeFactor) | **ECCV** 2022 |
| [Spatial Ensemble: a Novel Model Smoothing Mechanism for Student-Teacher Framework](https://proceedings.neurips.cc/paper/2021/hash/8597a6cfa74defcbde3047c891d78f90-Abstract.html) | [Spatial_Ensemble](https://github.com/tengteng95/Spatial_Ensemble) | **NeurIPS** 2021 |
| [Collaboration by Competition: Self-coordinated Knowledge Amalgamation for Multi-talent Student Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510630.pdf) |  | **ECCV** 2020 |
| [Multiple Expert Brainstorming for Domain Adaptive Person Re-identification](https://arxiv.org/abs/2007.01546) | [MEB-Net](https://github.com/YunpengZhai/MEB-Net) | **ECCV** 2020 |
| [Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Data-Free_Knowledge_Amalgamation_via_Group-Stack_Dual-GAN_CVPR_2020_paper.pdf) |          | **CVPR** 2020 |
| [Customizing Student Networks From Heterogeneous Teachers via Adaptive Knowledge Amalgamation](https://arxiv.org/pdf/1908.07121.pdf) |          | **ICCV ** 2019 |
| [Amalgamating Filtered Knowledge: Learning Task-customized Student from Multi-task Teachers](https://arxiv.org/abs/1905.11569) |          | **IJCAI** 2019 |
| [Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning](http://arxiv.org/abs/1906.10546) | [code](https://github.com/zju-vipa/CommonFeatureLearning) | **IJCAI** 2019 |
| [Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More](https://arxiv.org/pdf/1904.10167.pdf) | [KAmalEngine](https://github.com/zju-vipa/KamalEngine) | **CVPR **2019 |
| [Amalgamating Knowledge towards Comprehensive Classification](https://arxiv.org/pdf/1811.02796.pdf) |          | **AAAI **2019 |



### 5.3 GNN

| **Paper Title**                                              | **Code** | **Pub & Date** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Amalgamating Knowledge From Heterogeneous Graph Neural Networks](https://openaccess.thecvf.com/content/CVPR2021/papers/Jing_Amalgamating_Knowledge_From_Heterogeneous_Graph_Neural_Networks_CVPR_2021_paper.pdf) | [ycjing](https://github.com/ycjing/AmalgamateGNN.PyTorch) | **CVPR **2021 |

## 6. Model Reassemble
| **Paper Title**                                              | **Code** | **Pub & Date** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Deep Model Reassembly](https://arxiv.org/pdf/2210.17409.pdf) | [DeRy](https://github.com/Adamdad/DeRy) | **NeurIPS** 2022 |
| [GAN Cocktail: Mixing GANs without Dataset Access](https://arxiv.org/pdf/2106.03847.pdf) | [GAN-cocktail](https://github.com/omriav/GAN-cocktail) | **ECCV **2022 |
|                                                              |          |                |
### 6.1 Model Evolution
| **Paper Title**                                              | **Code** | **Pub & Date** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Population-based evolutionary gaming for unsupervised person re-identification](https://arxiv.org/abs/2306.05236) |  | **IJCV **2023 |
|                                                              |          |                |
## 7. Others
### 7.1 External Data Retrieval

| **Paper Title**                                              | **Code** | **Pub & Date**  |
| ------------------------------------------------------------ | :------: | :-------------: |
| [Evaluating the External and Parametric Knowledge Fusion of Large Language Models](https://arxiv.org/pdf/2405.19010) |          | **ArXiv** 24.05 |
| [Knowledge Fusion and Semantic Knowledge Ranking for Open Domain Question Answering](https://arxiv.org/pdf/2004.03101) |          | **ArXiv** 20.04 |

### 7.2 Others
| **Paper Title**                                              | **Code** | **Pub & Date** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Adaptive Discovering and Merging for Incremental Novel Class Discovery](https://arxiv.org/abs/2403.03382) |          |  **AAAI **2024  |
|                                                              |          |                 |
| [Knowledge Fusion and Semantic Knowledge Ranking for Open Domain Question Answering](https://arxiv.org/pdf/2004.03101) |          | **ArXiv** 20.04 |
### 7.2 Other Surveys
| **Paper Title**                                              | **Code** | **Pub & Date** |
| ------------------------------------------------------------ | :----: | :------------: |
| [Deep Model Fusion: A Survey](https://arxiv.org/abs/2309.15698) |  | **ArXiv** 23.09 |
| [Arcee's MergeKit: A Toolkit for Merging Large Language Models](https://arxiv.org/abs/2403.13257) |     [MergeKit](https://github.com/arcee-ai/mergekit)      | **ArXiv** 24.03 |
| A curated paper list of Model Merging methods | [ycjing](https://github.com/ycjing/Awesome-Model-Merging) | GitHub |
----------

## Contributors

Junlin Lee

Qi Tang

Runhua Jiang

**Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=duguodong7/Awesome-Knowledge-Fusion&type=Date)](https://star-history.com/#duguodong7/Awesome-Knowledge-Fusion&Date)

----------


## Contact
<!-- **Contact** -->

We invite all researchers to contribute to this repository, **'Knowledge Fusion: The Integration of Model Capabilities'**.
If you have any questions about the library, please feel free to contact us.

Email: duguodong7@gmail.com
