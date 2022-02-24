

# Awesome Dialogue State Tracking

**Dialogue State Tracking (DST) Papers, Codes, Datasets, Resources**

*✅ &nbsp;Last update : 22.01.13*

###
###

### [Table of Contents]

[📖&nbsp; Introduction to DST](#1-introduction-to-dst)

[📝&nbsp; DST Research Papers](#2-dst-research-papers)

&nbsp;&ensp;&emsp;[1. MultiWOZ (Multi-domain Wizard-of-Oz)](#1-multiwoz-multi-domain-wizard-of-oz)

&nbsp;&ensp;&emsp;&nbsp;&ensp;&emsp;[1) Ontology based model](#1-ontology-based-model)

&nbsp;&ensp;&emsp;&nbsp;&ensp;&emsp;[2) Open vocab based model](#2-open-vocab-based-model)

&nbsp;&ensp;&emsp;&nbsp;&ensp;&emsp;[3) Hybrid model (Ontology + Open vocab)](#3-Hybrid-model-ontology-open-vocab)

&nbsp;&ensp;&emsp;&nbsp;&ensp;&emsp;[4) Zero,Few-Shot / Meta / Transfer learning](#4-zero-few-shot-meta-transfer-learning)

&nbsp;&ensp;&emsp;[2. WOZ (Wizard-of-Oz)](#2-woz-wizard-of-oz)

&nbsp;&ensp;&emsp;[3. SGD (Schema-Guided Dialogue)](#3-sgd-schema-guided-dialogue)

&nbsp;&ensp;&emsp;[4. Data Limitation](#4-data-limitation)

&nbsp;&ensp;&emsp;[5. etc](#5-etc)

[🗂&nbsp; Datasets](#-3-datasets)

&nbsp;&ensp;&emsp;[1. Single Domain](#1-single-domain)

&nbsp;&ensp;&emsp;[2. Multi Domain](#2-multi-domain)

&nbsp;&ensp;&emsp;&nbsp;&ensp;&emsp;[English](#english)

&nbsp;&ensp;&emsp;&nbsp;&ensp;&emsp;[Korean](#korean)

&nbsp;&ensp;&emsp;&nbsp;&ensp;&emsp;[Chinese](#chinese)

[📌&nbsp; Evaluation Metrics](#4-evaluation-metrics)

[🏆&nbsp; Competition](#5-competition-dstc)



---

## [1] Introduction to DST


![img1](https://github.com/yukyunglee/Awesome-Dialogue-State-Tracking/blob/6d1a4f5bd2dc619c8dac08138182c92bb900730d/Img/%231.png)

**Dialogue state tracking (DST)** is a core component in task-oriented dialogue systems, such as restaurant reservation or ticket booking. The goal of DST is to **extract user goals/intentions expressed during conversation** and **to encode them as a compact set of dialogue states**, i.e., a set of slots and their corresponding values (Wu et al., 2019)

![img2](https://github.com/yukyunglee/Awesome-Dialogue-State-Tracking/blob/6d1a4f5bd2dc619c8dac08138182c92bb900730d/Img/%232.png)

**Dialogue State Tracking (DST)** can be **categorized into several approaches**. In this repository, we divided the dst approach as shown.



## [2] DST Research Papers


✅&nbsp; **Paper name, Venue | Model name | [Code]**

###
### 1. MultiWOZ (Multi-domain Wizard-of-Oz)

#### 1) Ontology based model

* *[SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking](https://arxiv.org/pdf/1907.07421.pdf)* , ACL 2019 | **SUMBT** | [[Code](https://github.com/SKTBrain/SUMBT)]
* *[HyST: A Hybrid Approach for Flexible and Accurate Dialogue State Tracking](https://arxiv.org/pdf/1907.00883.pdf)* , Interspeech 2019 | **HyST** | `None`
* *[Multi-domain dialogue state tracking as dynamic knowledge graph enhanced question answering](https://arxiv.org/pdf/1911.06192.pdf)* , arXiv preprint| **DSTQA** | [[Code](https://github.com/alexa/dstqa)]
* *[Schema-Guided Multi-Domain Dialogue State Tracking with Graph Attention Neural Networks](https://speechlab.sjtu.edu.cn/papers/2020/lc918-chen-aaai20.pdf)* , AAAI 2020 | **SST** | `None`
* *[Slot Self-Attentive Dialogue State Tracking](https://arxiv.org/pdf/2101.09374.pdf)* , WWW 2021 | **DST-STAR** | [[Code](https://github.com/smartyfh/DST-STAR)]
* *[Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking](https://arxiv.org/pdf/2104.04466.pdf)* , EMNLP 2021 |  `None` | [[Code](https://github.com/LinWeizheDragon/Knowledge-Aware-Graph-Enhanced-GPT-2-for-Dialogue-State-Tracking)]



#### 2) Open vocab based model

* *[Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/1905.08743.pdf)* , ACL 2019 | **TRADE** | [[Code](https://github.com/jasonwu0731/trade-dst)]

* *[BERT-DST: Scalable End-to-End Dialogue State Tracking with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1907.03040.pdf)*  , Interspeech 2019 | **BERT-DST** |[[Code](https://github.com/guanlinchao/bert-dst)]

* *[Scalable and Accurate Dialogue State Tracking via Hierarchical Sequence Generation](https://arxiv.org/pdf/1909.00754v2.pdf)* , IJCNLP 2019 | **COMER** | [[Code](https://github.com/renll/ComerNet)]

* *[CREDIT: Coarse-to-Fine Sequence Generation for Dialogue State Tracking](https://arxiv.org/pdf/2009.10435.pdf)* , arXiv preprint 2020 | **CREDIT** | `None`

* *[Non-Autoregressive Dialog State Tracking](https://openreview.net/pdf?id=H1e_cC4twS)* , ICLR 2020 | **NADST** | [[Code](https://github.com/henryhungle/NADST)]

* *[SimpleTOD: A Simple Language Model for Task-Oriented Dialogue](https://arxiv.org/pdf/2005.00796.pdf)* , NeurIPS 2020 | **SimpleTOD** | [[Code](https://github.com/salesforce/simpletod)]

* *[SAS: Dialogue State Tracking via Slot Attention and Slot Information Sharing](https://www.aclweb.org/anthology/2020.acl-main.567.pdf)* , ACL 2020 | **SAS** | `None`

* *[From Machine Reading Comprehension to Dialogue State Tracking: Bridging the Gap](https://arxiv.org/pdf/2004.05827.pdf)* , ACL 2020 | **STARC** | `None`

* *[Efficient Dialogue State Tracking by Selectively Overwriting Memory](https://arxiv.org/pdf/1911.03906.pdf)* , ACL 2020 | **SOM-DST** | [[Code](https://github.com/clovaai/som-dst)]

* *[End-to-End Neural Pipeline for Goal-Oriented Dialogue Systems using GPT-2](https://aclanthology.org/2020.acl-main.54.pdf)* , ACL 2020 | **NP-DST** | `None` 

* *[Efficient Context and Schema Fusion Networks for Multi-Domain Dialogue State Tracking](https://arxiv.org/pdf/2004.03386v4.pdf)* , Findings of EMNLP 2020 | **CSFN-DST** | `None`

* *[Multi-Domain Dialogue State Tracking based on State Graph](https://arxiv.org/pdf/2010.11137.pdf)* , arXiv preprint | **Graph-DST** | `None`

* *[GCDST: A Graph-based and Copy-augmented Multi-domain Dialogue State Tracking](https://www.aclweb.org/anthology/2020.findings-emnlp.95.pdf)* , Findings of EMNLP 2020 | **GCDST** | `None`

* *[Slot Attention with Value Normalization for Multi-Domain Dialogue State Tracking](https://www.aclweb.org/anthology/2020.emnlp-main.151.pdf)* , ACL 2020 | **SAVN** | [[Code](https://github.com/wyxlzsq/savn)]

* *[Parallel Interactive Networks for Multi-Domain Dialogue State Generation](https://www.aclweb.org/anthology/2020.emnlp-main.151.pdf)* , EMNLP 2020 | **PIN** | [[Code](https://github.com/zengyan-97/Transformer-DST)]

* *[TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue](https://arxiv.org/pdf/2004.06871.pdf)* , EMNLP 2020 | **TOD-BERT** | [[Code](https://github.com/jasonwu0731/ToD-BERT)]

* *[TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/pdf/2005.02877.pdf)* , SIGDAL 2020 | **TripPy** | [[Code](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public)]

* *[A Sequence-to-Sequence Approach to Dialogue State Tracking](https://arxiv.org/pdf/2011.09553.pdf)* , ACL 2021 | **Seq2Seq-DU** | [[Code](https://github.com/sweetalyssum/Seq2Seq-DU)]

* *[Jointly Optimizing State Operation Prediction and Value Generation for Dialogue State Tracking](https://arxiv.org/pdf/2010.14061.pdf)* , arXiv preprint | **Transformer-DST** | [[Code](https://github.com/zengyan-97/Transformer-DST)]

  

#### 3) Hybrid model (Ontology + Open vocab)

* *[Find or Classify? Dual Strategy for Slot-Value Predictions on Multi-Domain Dialog State Tracking](https://arxiv.org/pdf/1910.03544.pdf)* , SEM 2020 | **DS-DST** | `None`

*  *[Dual Slot Selector via Local Reliability Verification for Dialogue State Tracking](https://arxiv.org/pdf/2107.12578.pdf)* , ACL 2021 | **DSS-DST** | [[Code](https://github.com/guojinyu88/DSSDST)]



#### 4) Zero,Few-Shot / Meta / Transfer learning

* *[Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/1905.08743.pdf)* , ACL 2019 | **TRADE** | [[Code](https://github.com/jasonwu0731/trade-dst)]
* *[Fine-Tuning BERT for Schema-Guided Zero-Shot Dialogue State Tracking](https://arxiv.org/pdf/2002.00181.pdf)* , AAAI 2020 | **SGP-DST** | `None`
* *[Zero-Shot Transfer Learning with Synthesized Data for Multi-Domain Dialogue State Tracking](https://www.aclweb.org/anthology/2020.acl-main.12.pdf)* , ACL 2020 | `None` | [[Code](https://github.com/stanford-oval/zero-shot-multiwoz-acl2020)]
* *[From Machine Reading Comprehension to Dialogue State Tracking: Bridging the Gap](https://arxiv.org/pdf/2004.05827.pdf)* , ACL 2020 | `None` |  `None`
* *[MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/2009.12005.pdf)* , EMNLP 2020 | **MinTL** | [[Code](https://github.com/zlinao/MinTL)] 
* *[Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking](https://www.aclweb.org/anthology/2021.naacl-main.448.pdf)* , NAACL 2021 | `None` | `None`
* *[Zero-shot Generalization in Dialog State Tracking through Generative Question Answering](https://www.aclweb.org/anthology/2021.eacl-main.91.pdf)* , EACL 2021 | `None` | `None`
* *[Few Shot Dialogue State Tracking using Meta-learning](https://www.aclweb.org/anthology/2021.eacl-main.148.pdf)* , EACL 2021 | `None` | `None`
* *[Domain Adaptive Meta-learning for Dialogue State Tracking](https://ieeexplore.ieee.org/abstract/document/9431715)* , TASLP | **DAMAML** | [[Code](https://github.com/DeepLearnXMU/DAMAML)]
* *[Preview, Attend and Review: Schema-Aware Curriculum Learning for Multi-Domain Dialog State Tracking](https://arxiv.org/pdf/2106.00291.pdf)* , ACL 2021 | **ScCLog** | `None`
* *[NeuralWOZ: Learning to Collect Task-Oriented Dialogue via Model-Based Simulation](https://arxiv.org/pdf/2105.14454.pdf)* , ACL 2021 | **NeuralWOZ** | [[Code](https://github.com/naver-ai/neuralwoz)]


###
### 2. WoZ (Wizard-of-Oz)


* *[Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://arxiv.org/pdf/1606.03777v2.pdf)* , ACL 2017 | `None` | `None`

* *[Towards Universal Dialogue State Tracking](https://arxiv.org/pdf/1810.09587v1.pdf)* , EMNLP 2018 | **StateNet** | [[Code](https://github.com/renll/StateNet)]

* *[Toward Scalable Neural Dialogue State Tracking](https://arxiv.org/pdf/1812.00899.pdf)* , NeurIPS 2018 | **GCE** | [[Code](https://github.com/elnaaz/GCE-Model)]

* *[Global-Locally Self-Attentive Dialogue State Tracker](https://arxiv.org/pdf/1805.09655v3.pdf)* , ACL 2018 | **GLAD** | [[Code](https://github.com/salesforce/glad)]

* *[Scalable Neural Dialogue State Tracking](https://arxiv.org/pdf/1910.09942.pdf)* , ASRU 2019 | **G-SAT** | [[Code](https://github.com/vevake/GSAT)]

* *[A Simple but Effective BERT Model for Dialog State Tracking on Resource-Limited Systems](https://arxiv.org/pdf/1910.12995.pdf)* , ICASSP 2020 | `None` | `None`

* *[MA-DST: Multi-Attention-Based Scalable Dialog State Tracking](https://arxiv.org/pdf/2002.08898.pdf)* , AAAI 2020 | **MA-DST** | `None`

* *[TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue](https://arxiv.org/pdf/2004.06871.pdf)* , EMNLP 2020 | **TOD-BERT** | [[Code](https://github.com/jasonwu0731/ToD-BERT)]

* *[Neural Dialogue State Tracking with Temporally Expressive Networks](https://aclanthology.org/2020.findings-emnlp.142.pdf)* , Findings of EMNLP 2020 | **TEN** | [[Code](https://github.com/BDBC-KG-NLP/TEN_EMNLP2020)]

* *[A Sequence-to-Sequence Approach to Dialogue State Tracking](https://arxiv.org/pdf/2011.09553.pdf)* , ACL 2021 | **Seq2Seq-DU** | [[Code](https://github.com/sweetalyssum/Seq2Seq-DU)]


###
### 3. SGD (Schema-Guided Dialogue)

* *[A Fast and Robust BERT-based Dialogue State Tracker for Schema-Guided Dialogue Dataset](https://arxiv.org/pdf/2008.12335.pdf)* , KDD 2020 | **FastSGT** | [[Code](https://github.com/NVIDIA/NeMo)]

* *[A Sequence-to-Sequence Approach to Dialogue State Tracking](https://arxiv.org/pdf/2011.09553.pdf)* , ACL 2021 | **Seq2Seq-DU** | [[Code](https://github.com/sweetalyssum/Seq2Seq-DU)]



###

### 4. Data Limitation

* *[COCO: CONTROLLABLE COUNTERFACTUALS FOR EVALUATING DIALOGUE STATE TRACKERS](https://arxiv.org/pdf/2010.12850.pdf)* , ICLR 2021 | **CoCo** | [[Code](https://github.com/salesforce/coco-dst)]

* *[Annotation Inconsistency and Entity Bias in MultiWOZ](https://arxiv.org/pdf/2105.14150.pdf)* , SIGDIAL 2021 | `None`  | `None` 

* *[Oh My Mistake!: Toward Realistic Dialogue State Tracking including Turnback Utterances](https://arxiv.org/pdf/2108.12637.pdf)* , arXiv preprint | `None`  | `None` 

  
  
  

###

### 5. etc.

* *[Recent Advances and Challenges in Task-oriented Dialog Systems](https://arxiv.org/pdf/2003.07490.pdf)* , SCTS | `None` | `None`
* *[Variational Hierarchical Dialog Autoencoder for Dialog State Tracking Data Augmentation](https://www.aclweb.org/anthology/2020.emnlp-main.274.pdf)* , EMNLP 2020 | `None` | [[Code](https://github.com/kaniblu/vhda)]
* *[Tutorial : Deeper Conversational AI](https://neurips.cc/media/Slides/nips/2020/virtual(07-08-00)-07-08-00UTC-16657-track2_deeper.pdf)* , NeurIPS 2020 | `None` | `None` 
* *[Out-of-Task Training for Dialog State Tracking Models](https://aclanthology.org/2020.coling-main.596.pdf)* , COLING 2020 | **Trippy** variant |[[Code](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public)]
* *[DialoGLUE: A Natural Language Understanding Benchmark for Task-Oriented Dialogue](https://arxiv.org/pdf/2009.13570.pdf)* , arXiv preprint | **DialoGLUE** | [[Code](https://github.com/alexa/dialoglue)]
* *[A Comparative Study on Schema-Guided Dialogue State Tracking](https://www.aclweb.org/anthology/2021.naacl-main.62.pdf)* , NAACL 2021 | `None` | `None`
* *[Comprehensive Study: How the Context Information of Different Granularity Affects Dialogue State Tracking?](https://arxiv.org/pdf/2105.03571.pdf)* , ACL 2021 | `None` | [[Code](https://github.com/yangpuhai/Granularity-in-DST)]
* *[Preview, Attend and Review: Schema-Aware Curriculum Learning for Multi-Domain Dialog State Tracking](https://arxiv.org/pdf/2106.00291.pdf)* , ACL 2021 |  **SaCLog** | `None`
* *[Coreference Augmentation for Multi-Domain Task-Oriented Dialogue State Tracking](https://arxiv.org/pdf/2106.08723.pdf)* , Interspeech 2021 | **CDST** | `None`




###
## [3] Datasets

✅&nbsp; **Paper name, Venue | Dataset name | Language | [Code]**

### 1. Single Domain

* *[The Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W13-4065.pdf)* , SIGDIAL 2013 | **DSTC** | en | [[Dataset](https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/#!dstc1-downloads)]

* *[The Second Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W14-4337.pdf)* , SIGDIAL 2014 | **DSTC2** | en | [[Dataset](https://github.com/matthen/dstc)]

* *[A Network-based End-to-End Trainable Task-oriented Dialogue System](https://www.aclweb.org/anthology/E17-1042.pdf)* , EACL 2017 | **CamRest/CamRest676** | en | [[Dataset](https://github.com/WING-NUS/sequicity/tree/master/data/CamRest676)]

* *[Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://www.aclweb.org/anthology/P17-1163.pdf)* , ACL 2017 | **WOZ 2.0** | en, de, it | [[Dataset](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz)]

* *[AirDialogue: An Environment for Goal-Oriented Dialogue Research](https://www.aclweb.org/anthology/D18-1419.pdf)* , EMNLP 2018 | **AirDialogue** | en | [[Dataset](https://github.com/google/airdialogue)]

* *[HDRS: Hindi Dialogue Restaurant Search Corpus for Dialogue State Tracking in Task-Oriented Environment](https://ieeexplore.ieee.org/document/9376978)* , TASLP | **HDRS** | hi | [[Dataset](https://github.com/skmalviya/HDRS-Corpus)]

###
### 2. Multi Domain

#### English

* *[The Third Dialog State Tracking Challenge](https://www.matthen.com/assets/pdf/The_Third_Dialog_State_Tracking_Challenge.pdf)* , IEEE SLT 2014 | **DSTC3** | en | [[Dataset](https://github.com/matthen/dstc)]

* *[Key-Value Retrieval Networks for Task-Oriented Dialogue](https://www.aclweb.org/anthology/W17-5506.pdf)* , SIGDIAL 2017 | **KVReT** | en | [[Dataset](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)]

* *[Building a Conversational Agent Overnight with Dialogue Self-Play](https://arxiv.org/pdf/1801.04871.pdf)* , arXiv preprint | **SimD** | en | [[Dataset](https://github.com/google-research-datasets/simulated-dialogue)]

* *[Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing](https://www.aclweb.org/anthology/P18-2069.pdf)* , ACL 2018 | **MultiWOZ 1.0** | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

* *[MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://www.aclweb.org/anthology/D18-1547.pdf)* , EMNLP 2018 | **MultiWOZ 2.0** | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

* *[MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines](https://www.aclweb.org/anthology/2020.lrec-1.53.pdf)* , LREC 2020 | **MultiWOZ 2.1** | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

* *[Schema-Guided Dialogue State Tracking Task at DSTC8](https://arxiv.org/pdf/2002.01359.pdf)* , AAAI 2020 | **SGD** | en | [[Dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)]

* *[MultiWOZ 2.2 : A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://www.aclweb.org/anthology/2020.nlp4convai-1.13.pdf)* , ACL 2020 | **MultiWOZ 2.2** | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

* *[MultiWOZ 2.3: A multi-domain task-oriented dialogue dataset enhanced with annotation corrections and co-reference annotation](https://arxiv.org/pdf/2010.05594.pdf)* , arXiv preprint | **MultiWOZ 2.3** | en | [[Dataset](https://github.com/lexmen318/MultiWOZ-coref)]

* *[MultiWOZ 2.4: A Multi-Domain Task-Oriented Dialogue Dataset with Essential Annotation Corrections to Improve State Tracking Evaluation](https://arxiv.org/pdf/2104.00773.pdf)* , arXiv preprint | **MultiWOZ 2.4** | en | [[Dataset](https://github.com/smartyfh/MultiWOZ2.4)]

* *[BiToD: A Bilingual Multi-Domain Dataset For Task-Oriented Dialogue Modeling](https://arxiv.org/pdf/2106.02787.pdf)* , NeurIPS 2021 Dataset and Benchmark Track | **BiToD** | en, ch | [[Dataset](https://github.com/HLTCHKUST/BiToD)]

#### Korean 

* *[KLUE: Korean Language Understanding Evaluation](https://arxiv.org/pdf/2105.09680.pdf)* , NeurIPS 2021 Dataset and Benchmark Track | **KLUE-DST/WoS** | kr | [[Dataset](https://klue-benchmark.com/)]

#### Chinese

* *[CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://www.aclweb.org/anthology/2020.tacl-1.19.pdf)* , TACL | **CrossWOZ** | ch | [[Dataset](https://github.com/thu-coai/CrossWOZ)]

* *[RiSAWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset with Rich Semantic Annotations for Task-Oriented Dialogue Modeling](https://aclanthology.org/2020.emnlp-main.67.pdf)* , EMNLP 2020 | **RiSAWOZ** | ch | [[Dataset](https://github.com/terryqj0107/RiSAWOZ)]

* *[BiToD: A Bilingual Multi-Domain Dataset For Task-Oriented Dialogue Modeling](https://arxiv.org/pdf/2106.02787.pdf)* , NeurIPS 2021 Dataset and Benchmark Track | **BiToD** | en, ch | [[Dataset](https://github.com/HLTCHKUST/BiToD)]

###

## [4] Evaluation Metrics

✅&nbsp; **Paper name, Venue | Metric name**

* *[Global-Locally Self-Attentive Dialogue State Tracker](https://aclanthology.org/P18-1135.pdf)* , ACL 2018 | **Joint Goal Accuracy** 

* *[Schema-Guided Dialogue State Tracking Task at DSTC8](https://arxiv.org/pdf/2002.01359.pdf)* , AAAI 2020 | **Average Goal Accuracy** 

* *Mismatch between Multi-turn Dialogue and its Evaluation Metric in Dialogue State Tracking*, ACL 2022 | **Relative Slot Accuracy**

###

## [5] Competition (DSTC)

### 1. Introduction

**DSTC** is the most famous competition in the field of Dialogue System. First held in 2013, DSTC started as a **Dialogue State Tracking Challenge**, but since the dialogue-related researches have been actively expanded, it has been relaunched as the **Dialogue System Technology Challenges**. DSTC covers the various subjects of dialogue issues such as NLP, Vision, and Speech. The 10th challenge is now taking place with a total of 5 tracks. More information about DSTC can be found at the [link](https://sites.google.com/dstc.community/dstc10/).



### 2. Related Papers

✅&nbsp; **Paper name, Competition | Model name | [Code]**

* *[An Empirical Study of Cross-Lingual Transferability in Generative Dialogue State Tracker](https://arxiv.org/pdf/2101.11360.pdf)* , DSTC9 Workshop - AAAI  | `None` | `None`

* *[Efficient Dialogue State Tracking by Masked Hierarchical Transformer](https://arxiv.org/pdf/2106.14433.pdf)* , DSTC9 Workshop - AAAI | `None` | `None`




###
#### 💌&nbsp; Contact Us

> **Yukyung Lee | Korea University | yukyung_lee@korea.ac.kr**

> **Kyumin Park | Korea Advanced Institute of Science and Technology | pkm9403@gmail.com**

