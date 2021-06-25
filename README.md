

# Awesome Dialogue State Tracking

Dialogue State Tracking (DST) Papers, Codes, Datasets, Resources

(정리할 자료 : 논문(논문이름,링크,Venue) , 코드링크 , 데이터셋 링크)


## Introduction to DST

![img](https://github.com/yukyunglee/Awesome-Dialogue-State-Tracking/blob/355ea9c95b3765221ddbdcceab251926865a3021/Img/introduction_dst.png?raw=true)

**Dialogue state tracking (DST)** is a core component in task-oriented dialogue systems, such as restaurant reservation or ticket booking. The goal of DST is to **extract user goals/intentions expressed during conversation** and **to encode them as a compact set of dialogue states**, i.e., a set of slots and their corresponding values (Wu et al., 2019)



## DST Research Papers



(이런식으로 적으면 어떨까요 !?)

✅ pdf로 바로 연결되는 링크로 !

✅ 출간 년도 순서대로 sorting하는게 좋아보여요 !



논문이름, venue | 모델 이름 | [Code]



### 1. Ontology based model

​	*[SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking](https://arxiv.org/pdf/1907.07421.pdf)* , ACL 2019 | **SUMBT** | [[Code](https://github.com/SKTBrain/SUMBT)]

​	*[HyST: A Hybrid Approach for Flexible and Accurate Dialogue State Tracking](https://arxiv.org/pdf/1907.00883.pdf)* , Interspeech 2019 | **HyST** | `None`

​	*[Multi-domain dialogue state tracking as dynamic knowledge graph enhanced question answering](https://arxiv.org/pdf/1911.06192.pdf)* , arxiv 2019| **DSTQA** | [[Code](https://github.com/alexa/dstqa)]

​	*[Schema-Guided Multi-Domain Dialogue State Tracking with Graph Attention Neural Networks](https://speechlab.sjtu.edu.cn/papers/2020/lc918-chen-aaai20.pdf)* , AAAI 2020 | **SST** | `None`

​	*[A Contextual Hierarchical Attention Network with Adaptive Objective for Dialogue State Tracking](https://www.aclweb.org/anthology/2020.acl-main.563.pdf)* , ACL 2020 | **CHAN-DST** | [[Code](https://github.com/smartyfh/CHAN-DST)]

​	*[Slot Self-Attentive Dialogue State Tracking](https://arxiv.org/pdf/2101.09374.pdf)* , WWW 2021 | **DST-STAR** | [[Code](https://github.com/smartyfh/DST-STAR)]



### 2. Open vocab based model

​	*[Efficient Dialogue State Tracking by Selectively Overwriting Memory](https://arxiv.org/pdf/1911.03906.pdf)* , ACL 2020 | **SOM-DST** | [[Code](https://github.com/clovaai/som-dst)]



### 3. Hybrid model (Ontology + Open vocab)



​	*[Find or Classify? Dual Strategy for Slot-Value Predictions on Multi-Domain Dialog State Tracking](https://arxiv.org/pdf/1910.03544.pdf)* , SEM 2020 | **DS-DST** | `None`





### 4. etc
​	*[COCO: CONTROLLABLE COUNTERFACTUALS FOR EVALUATING DIALOGUE STATE TRACKERS](https://arxiv.org/pdf/2010.12850.pdf)* , ICLR 2021 | CoCo | [[Code](https://github.com/salesforce/coco-dst)]

​	*[Zero-shot Generalization in Dialog State Tracking through Generative Question Answering](https://www.aclweb.org/anthology/2021.eacl-main.91.pdf)* , EACL 2021 | [[Code]]


## Datasets

### 1. Single Domain

​	*[The Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W13-4065.pdf)* , SIGDIAL 2013 | DSTC | en |[[Dataset](https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/#!dstc1-downloads)]

​	*[The Second Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W14-4337.pdf)* , SIGDIAL 2014 | DSTC2 | en | [[Dataset](https://github.com/matthen/dstc)]

​	*[A Network-based End-to-End Trainable Task-oriented Dialogue System](https://www.aclweb.org/anthology/E17-1042.pdf)* , EACL 2017 | CamRest/CamRest676 | en | [[Dataset](https://github.com/WING-NUS/sequicity/tree/master/data/CamRest676)]

​	*[Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://www.aclweb.org/anthology/P17-1163.pdf)* , ACL 2017 | WOZ 2.0 | en, de, it | [[Dataset](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz)]

​	*[AirDialogue: An Environment for Goal-Oriented Dialogue Research](https://www.aclweb.org/anthology/D18-1419.pdf)* , EMNLP 2018 | AirDialogue | en | [[Dataset](https://github.com/google/airdialogue)]

### 2. Multi Domain

#### English
​	*[The Third Dialog State Tracking Challenge](https://www.matthen.com/assets/pdf/The_Third_Dialog_State_Tracking_Challenge.pdf)* , IEEE SLT 2014 | DSTC3 | en | [[Dataset](https://github.com/matthen/dstc)]

​	*[Key-Value Retrieval Networks for Task-Oriented Dialogue](https://www.aclweb.org/anthology/W17-5506.pdf)* , SIGDIAL 2017 | KVReT | en | [[Dataset](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)]

  <!--- Dialogue state 없음 --->
​	*[Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems](https://www.aclweb.org/anthology/W17-5526v2.pdf)* , SIGDIAL 2017 | Frames | en | [[Dataset](https://www.microsoft.com/en-us/research/project/frames-dataset/)]

​	*[Building a Conversational Agent Overnight with Dialogue Self-Play](https://arxiv.org/pdf/1801.04871.pdf)* , arXiv preprint | SimD | en | [[Dataset](https://github.com/google-research-datasets/simulated-dialogue)]

​	*[Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing](https://www.aclweb.org/anthology/P18-2069.pdf)* , ACL 2018 | MultiWOZ 1.0 | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

​	*[MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://www.aclweb.org/anthology/D18-1547.pdf)* , EMNLP 2018 | MultiWOZ 2.0 | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

<!--- Dialogue state 없음 --->
​	*[MICROSOFT DIALOGUE CHALLENGE: BUILDING END-TO-END TASK-COMPLETION DIALOGUE SYSTEMS](https://arxiv.org/pdf/1807.11125.pdf)* , SLT 2018 | MDC | en | [[Dataset](https://github.com/xiul-msr/e2e_dialog_challenge)]

<!--- Dialogue state 없음 --->
​	*[CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases](https://www.aclweb.org/anthology/D19-1204.pdf)* , EMNLP 2019 | CoSQL | en | [[Dataset](https://yale-lily.github.io/cosql)]

<!--- Dialogue state 없음 --->
​	*[Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset](https://www.aclweb.org/anthology/D19-1459.pdf)* , EMNLP 2019 | Taskmaster-1 | en | [[Dataset](https://research.google/tools/datasets/taskmaster-1/)]

​	*[MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines](https://www.aclweb.org/anthology/2020.lrec-1.53.pdf)* , LREC 2020 | MultiWOZ 2.1 | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

​	*[Schema-Guided Dialogue State Tracking Task at DSTC8](https://arxiv.org/pdf/2002.01359.pdf)* , AAAI 2020 | SGD | en | [[Dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)]

​	*[MultiWOZ 2.2 : A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://www.aclweb.org/anthology/2020.nlp4convai-1.13.pdf)* , ACL 2020 | MultiWOZ 2.2 | en | [[Dataset](https://github.com/budzianowski/multiwoz)]

​	*[MultiWOZ 2.3: A multi-domain task-oriented dialogue dataset enhanced with annotation corrections and co-reference annotation](https://arxiv.org/pdf/2010.05594.pdf)* , arXiv preprint | MultiWOZ 2.3 | en | [[Dataset](https://github.com/lexmen318/MultiWOZ-coref)]

​	*[MultiWOZ 2.4: A Multi-Domain Task-Oriented Dialogue Dataset with Essential Annotation Corrections to Improve State Tracking Evaluation](https://arxiv.org/pdf/2104.00773.pdf)* , arXiv preprint | MultiWOZ 2.4 | en | [[Dataset](https://github.com/smartyfh/MultiWOZ2.4)]


#### Korean
​	*[KLUE: Korean Language Understanding Evaluation](https://arxiv.org/pdf/2105.09680.pdf)* , arXiv preprint | KLUE-DST/WoS | kr | [[Dataset](https://klue-benchmark.com/)]

#### Chinese
​	*[CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://www.aclweb.org/anthology/2020.tacl-1.19.pdf)* , TACL | CrossWOZ | ch | [[Dataset](https://github.com/thu-coai/CrossWOZ)]





