

# Awesome Dialogue State Tracking

Dialogue State Tracking (DST) Papers, Codes, Datasets, Resources

(정리할 자료 : 논문(논문이름,링크,Venue) , 코드링크 , 데이터셋 링크)

Table of contents




## 📖 Introduction to DST

![img](https://github.com/yukyunglee/Awesome-Dialogue-State-Tracking/blob/9968749b84cb475e73369308e1e633148c765246/Img/intro_dst.png)

**Dialogue state tracking (DST)** is a core component in task-oriented dialogue systems, such as restaurant reservation or ticket booking. The goal of DST is to **extract user goals/intentions expressed during conversation** and **to encode them as a compact set of dialogue states**, i.e., a set of slots and their corresponding values (Wu et al., 2019)



## 📝 DST Research Papers



(이런식으로 적으면 어떨까요 !?)

✅ pdf로 바로 연결되는 링크로 ! 

✅ 출간 년도 순서대로 sorting하는게 좋아보여요 !

✅ Dataset - Model(5 categories) - YEAR Sorting



논문이름, venue | 모델 이름 | [Code]

### 1. MultiWOZ (Multi-domain Wizard-of-Oz)

####1) Ontology based model

* *[SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking](https://arxiv.org/pdf/1907.07421.pdf)* , ACL 2019 | **SUMBT** | [[Code](https://github.com/SKTBrain/SUMBT)]

* *[HyST: A Hybrid Approach for Flexible and Accurate Dialogue State Tracking](https://arxiv.org/pdf/1907.00883.pdf)* , Interspeech 2019 | **HyST** | `None`

* *[Multi-domain dialogue state tracking as dynamic knowledge graph enhanced question answering](https://arxiv.org/pdf/1911.06192.pdf)* , arXiv preprint| **DSTQA** | [[Code](https://github.com/alexa/dstqa)]

* *[Schema-Guided Multi-Domain Dialogue State Tracking with Graph Attention Neural Networks](https://speechlab.sjtu.edu.cn/papers/2020/lc918-chen-aaai20.pdf)* , AAAI 2020 | **SST** | `None`

* *[A Contextual Hierarchical Attention Network with Adaptive Objective for Dialogue State Tracking](https://www.aclweb.org/anthology/2020.acl-main.563.pdf)* , ACL 2020 | **CHAN-DST** | [[Code](https://github.com/smartyfh/CHAN-DST)]

* *[Slot Self-Attentive Dialogue State Tracking](https://arxiv.org/pdf/2101.09374.pdf)* , WWW 2021 | **DST-STAR** | [[Code](https://github.com/smartyfh/DST-STAR)]



####2) Open vocab based model

* *[Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/1905.08743.pdf)* , ACL 2019 | **TRADE** | [[Code](https://github.com/jasonwu0731/trade-dst)]

* *[BERT-DST: Scalable End-to-End Dialogue State Tracking with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1907.03040.pdf)*  , Interspeech 2019 | **BERT-DST** |[[Code](https://github.com/guanlinchao/bert-dst)]

* *[Scalable and Accurate Dialogue State Tracking via Hierarchical Sequence Generation](https://arxiv.org/pdf/1909.00754v2.pdf)* , IJCNLP 2019 | **COMER** | [[Code](https://github.com/renll/ComerNet)]

* [CREDIT: Coarse-to-Fine Sequence Generation for Dialogue State Tracking](https://arxiv.org/pdf/2009.10435.pdf) , arXiv preprint 2020 | **CREDIT** | `None`

* *[Non-Autoregressive Dialog State Tracking](https://openreview.net/pdf?id=H1e_cC4twS)* , ICLR 2020 | **NADST** | [[Code](https://github.com/henryhungle/NADST)]

* *[SimpleTOD: A Simple Language Model for Task-Oriented Dialogue](https://arxiv.org/pdf/2005.00796.pdf)* , NeurIPS 2020 | **SimpleTOD** | [[Code](https://github.com/salesforce/simpletod)]

* *[SAS: Dialogue State Tracking via Slot Attention and Slot Information Sharing](https://www.aclweb.org/anthology/2020.acl-main.567.pdf)* , ACL 2020 | **SAS** | `None`

* [From Machine Reading Comprehension to Dialogue State Tracking: Bridging the Gap](https://arxiv.org/pdf/2004.05827.pdf) , ACL 2020 | **STARC ** | `None`

* *[Efficient Dialogue State Tracking by Selectively Overwriting Memory](https://arxiv.org/pdf/1911.03906.pdf)* , ACL 2020 | **SOM-DST** | [[Code](https://github.com/clovaai/som-dst)]

* [End-to-End Neural Pipeline for Goal-Oriented Dialogue Systems using GPT-2](https://aclanthology.org/2020.acl-main.54.pdf) , ACL 2020 | **NP-DST** | `None` 

* *[Efficient Context and Schema Fusion Networks for Multi-Domain Dialogue State Tracking](https://arxiv.org/pdf/2004.03386v4.pdf)* , Findings of ACL 2020 | **CSFN-DST** | `None`

* *[Multi-Domain Dialogue State Tracking based on State Graph](https://arxiv.org/pdf/2010.11137.pdf)* , arXiv preprint | **Graph-DST** | `None`

* *[GCDST: A Graph-based and Copy-augmented Multi-domain Dialogue State Tracking](https://www.aclweb.org/anthology/2020.findings-emnlp.95.pdf)* , Findings of ACL 2020 | **GCDST** | `None`

* *[Slot Attention with Value Normalization for Multi-Domain Dialogue State Tracking](https://www.aclweb.org/anthology/2020.emnlp-main.151.pdf)* , ACL 2020 | **SAVN** | [[Code](https://github.com/wyxlzsq/savn)]

* *[Parallel Interactive Networks for Multi-Domain Dialogue State Generation](https://www.aclweb.org/anthology/2020.emnlp-main.151.pdf)* , EMNLP 2020 | **PIN** | [[Code](https://github.com/zengyan-97/Transformer-DST)]

* *[TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue](https://arxiv.org/pdf/2004.06871.pdf)* , EMNLP 2020 | **TOD-BERT** | [[Code](https://github.com/jasonwu0731/ToD-BERT)]

* *[TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/pdf/2005.02877.pdf)* , SIGDAL 2020 | **TripPy** | [[Code](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public)]

* *[A Sequence-to-Sequence Approach to Dialogue State Tracking](https://arxiv.org/pdf/2011.09553.pdf)* , ACL 2021 | **Seq2Seq-DU** | [[Code](https://github.com/sweetalyssum/Seq2Seq-DU)]

* *[Jointly Optimizing State Operation Prediction and Value Generation for Dialogue State Tracking](https://arxiv.org/pdf/2010.14061.pdf)* , ACL 2021 | **Transformer-DST** | [[Code](https://github.com/zengyan-97/Transformer-DST)]

  

####3) Hybrid model (Ontology + Open vocab)

* *[Find or Classify? Dual Strategy for Slot-Value Predictions on Multi-Domain Dialog State Tracking](https://arxiv.org/pdf/1910.03544.pdf)* , SEM 2020 | **DS-DST** | `None`



#### 4) Zero,Few-Shot / Meta / Transfer learning

* *[Fine-Tuning BERT for Schema-Guided Zero-Shot Dialogue State Tracking](https://arxiv.org/pdf/2002.00181.pdf)* , AAAI 2020 | **SGP-DST** 

* *[Zero-Shot Transfer Learning with Synthesized Data for Multi-Domain Dialogue State Tracking](https://www.aclweb.org/anthology/2020.acl-main.12.pdf)* , ACL 2020 | [[Code](https://github.com/stanford-oval/zero-shot-multiwoz-acl2020)]
* *[MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/2009.12005.pdf)* , EMNLP 2020 | **MinTL** | [[Code](https://github.com/zlinao/MinTL)] 
* *[Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking](https://www.aclweb.org/anthology/2021.naacl-main.448.pdf)* , NAACL 2021

* *[Zero-shot Generalization in Dialog State Tracking through Generative Question Answering](https://www.aclweb.org/anthology/2021.eacl-main.91.pdf)* , EACL 2021

* *[Few Shot Dialogue State Tracking using Meta-learning](https://www.aclweb.org/anthology/2021.eacl-main.148.pdf)* , EACL 2021

* *[Domain Adaptive Meta-learning for Dialogue State Tracking](https://ieeexplore.ieee.org/abstract/document/9431715?casa_token=XXvG2B13qhgAAAAA:O-dhl2zV6dYNdLlffscdmJ9Ap6OBR2tYAPcxIWNpx9euy-Qb_1GJcDY7jZy5-o58x02VcHj0)* , TASLP | **DAMAML** | [[Code](https://github.com/DeepLearnXMU/DAMAML)]

  

####5) etc

* *[Recent Advances and Challenges in Task-oriented Dialog Systems](https://arxiv.org/pdf/2003.07490.pdf)* , SCTS

* *[Variational Hierarchical Dialog Autoencoder for Dialog State Tracking Data Augmentation](https://www.aclweb.org/anthology/2020.emnlp-main.274.pdf)* , EMNLP 2020 | [[Code](https://github.com/kaniblu/vhda)]

* *[COCO: CONTROLLABLE COUNTERFACTUALS FOR EVALUATING DIALOGUE STATE TRACKERS](https://arxiv.org/pdf/2010.12850.pdf)* , ICLR 2021 | **CoCo** | [[Code](https://github.com/salesforce/coco-dst)]

* *[A Comparative Study on Schema-Guided Dialogue State Tracking](https://www.aclweb.org/anthology/2021.naacl-main.62.pdf)* , NAACL 2021

* *[NeuralWOZ: Learning to Collect Task-Oriented Dialogue via Model-Based Simulation](https://arxiv.org/pdf/2105.14454.pdf)* , ACL 2021 | **NeuralWOZ** | [[Code](https://github.com/naver-ai/neuralwoz)]

* *[Coreference Augmentation for Multi-Domain Task-Oriented Dialogue State Tracking](https://arxiv.org/pdf/2106.08723.pdf)* , Interspeech 2021 | **CDST**



### 2. WoZ (Wizard-of-Oz)

* *[A Sequence-to-Sequence Approach to Dialogue State Tracking](https://arxiv.org/pdf/2011.09553.pdf)* , ACL 2021 | **Seq2Seq-DU** | [[Code](https://github.com/sweetalyssum/Seq2Seq-DU)]



### 3. SGD (Schema-Guided Dialogue)

* [A Fast and Robust BERT-based Dialogue State Tracker for Schema-Guided Dialogue Dataset](https://arxiv.org/pdf/2008.12335.pdf) , KDD 2020 | **FastSGT** | [[Code](https://github.com/NVIDIA/NeMo)]

* *[A Sequence-to-Sequence Approach to Dialogue State Tracking](https://arxiv.org/pdf/2011.09553.pdf)* , ACL 2021 | **Seq2Seq-DU** | [[Code](https://github.com/sweetalyssum/Seq2Seq-DU)]







## 🗂 Datasets

### 1. Single Domain

* *[The Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W13-4065.pdf)* , SIGDIAL 2013 | **DSTC** | en |[[Dataset](https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/#!dstc1-downloads)]

* *[The Second Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W14-4337.pdf)* , SIGDIAL 2014 | **DSTC2** | en | [[Dataset](https://github.com/matthen/dstc)]

* *[A Network-based End-to-End Trainable Task-oriented Dialogue System](https://www.aclweb.org/anthology/E17-1042.pdf)* , EACL 2017 | **CamRest/CamRest676** | en | [[Dataset](https://github.com/WING-NUS/sequicity/tree/master/data/CamRest676)]

* *[Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://www.aclweb.org/anthology/P17-1163.pdf)* , ACL 2017 | **WOZ 2.0** | en, de, it | [[Dataset](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz)]

* *[AirDialogue: An Environment for Goal-Oriented Dialogue Research](https://www.aclweb.org/anthology/D18-1419.pdf)* , EMNLP 2018 | **AirDialogue** | en | [[Dataset](https://github.com/google/airdialogue)]

### 2. Multi Domain

#### English

* *[The Third Dialog State Tracking Challenge](https://www.matthen.com/assets/pdf/The_Third_Dialog_State_Tracking_Challenge.pdf)* , IEEE SLT 2014 | **DSTC3** | en | [[Dataset](https://github.com/matthen/dstc)]

* *[Key-Value Retrieval Networks for Task-Oriented Dialogue](https://www.aclweb.org/anthology/W17-5506.pdf)* , SIGDIAL 2017 | **KVReT** | en | [[Dataset](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)]

  <!--- Dialogue state 없음 --->

* *[Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems](https://www.aclweb.org/anthology/W17-5526v2.pdf)* , SIGDIAL 2017 | **Frames** | en | [[Dataset](https://www.microsoft.com/en-us/research/project/frames-dataset/)]

* *[Building a Conversational Agent Overnight with Dialogue Self-Play](https://arxiv.org/pdf/1801.04871.pdf)* , arXiv preprint | **SimD** | en | [[Dataset](https://github.com/google-research-datasets/simulated-dialogue)]

* *[Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing](https://www.aclweb.org/anthology/P18-2069.pdf)* , ACL 2018 | **MultiWOZ 1.0** | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

* *[MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://www.aclweb.org/anthology/D18-1547.pdf)* , EMNLP 2018 | **MultiWOZ 2.0** | en | [[Dataset](https://github.com/budzianowski/multiwoz)]
  <!--- Dialogue state 없음 --->

* *[MICROSOFT DIALOGUE CHALLENGE: BUILDING END-TO-END TASK-COMPLETION DIALOGUE SYSTEMS](https://arxiv.org/pdf/1807.11125.pdf)* , SLT 2018 | **MDC** | en | [[Dataset](https://github.com/xiul-msr/e2e_dialog_challenge)]
  <!--- Dialogue state 없음 --->

* *[CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases](https://www.aclweb.org/anthology/D19-1204.pdf)* , EMNLP 2019 | **CoSQL** | en | [[Dataset](https://yale-lily.github.io/cosql)]
  <!--- Dialogue state 없음 --->

* *[Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset](https://www.aclweb.org/anthology/D19-1459.pdf)* , EMNLP 2019 | **Taskmaster-1** | en | [[Dataset](https://research.google/tools/datasets/taskmaster-1/)]

* *[MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines](https://www.aclweb.org/anthology/2020.lrec-1.53.pdf)* , LREC 2020 | **MultiWOZ 2.1** | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

* *[Schema-Guided Dialogue State Tracking Task at DSTC8](https://arxiv.org/pdf/2002.01359.pdf)* , AAAI 2020 | **SGD** | en | [[Dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)]

* *[MultiWOZ 2.2 : A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://www.aclweb.org/anthology/2020.nlp4convai-1.13.pdf)* , ACL 2020 | **MultiWOZ 2.2** | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

* *[MultiWOZ 2.3: A multi-domain task-oriented dialogue dataset enhanced with annotation corrections and co-reference annotation](https://arxiv.org/pdf/2010.05594.pdf)* , arXiv preprint | **MultiWOZ 2.3** | en | [[Dataset](https://github.com/lexmen318/MultiWOZ-coref)]

* *[MultiWOZ 2.4: A Multi-Domain Task-Oriented Dialogue Dataset with Essential Annotation Corrections to Improve State Tracking Evaluation](https://arxiv.org/pdf/2104.00773.pdf)* , arXiv preprint | **MultiWOZ 2.4** | en | [[Dataset](https://github.com/smartyfh/MultiWOZ2.4)]


#### Korean

* *[KLUE: Korean Language Understanding Evaluation](https://arxiv.org/pdf/2105.09680.pdf)* , arXiv preprint | **KLUE-DST/WoS** | kr | [[Dataset](https://klue-benchmark.com/)]

#### Chinese

* *[CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://www.aclweb.org/anthology/2020.tacl-1.19.pdf)* , TACL | **CrossWOZ** | ch | [[Dataset](https://github.com/thu-coai/CrossWOZ)]



## 🏆 Competition

### 1. Introduction



### 2. Papers

* *[An Empirical Study of Cross-Lingual Transferability in Generative Dialogue State Tracker](https://arxiv.org/pdf/2101.11360.pdf)* , DSTC9 Workshop - AAAI  | `None`

