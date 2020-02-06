---
title: "Relation Classification via Convolutional Deep Neural Network"
layout: post
date: 2018-02-19
headerImage: false
tag:
- convolutional-neural-network
- relation-extraction
- relation-classification
category: blog
author: roomylee
---

- Paper Link: <http://www.aclweb.org/anthology/C14-1220>
- Author
  - Daojian Zeng (Chinese Academy of Sciences)
  - Kang Liu (Chinese Academy of Sciences)
  - Siwei Lai (Chinese Academy of Sciences)
  - Guangyou Zhou (Chinese Academy of Sciences)
  - Jun Zhao (Chinese Academy of Sciences)
- Published at
  - COLING 2014

---

## Abstract

- 이 분야의 state of the art는 확률적 머신러닝 기법이고 이는 feature 추출의 질이 성능을 좌지우지함
- Deep CNN을 사용하여 lexical & sentence level feature를 추출하려고 함
- 우리는 복잡한 전처리가 필요하지 않음
- 먼저 word embedding lookup table을 통해 word 토큰을 벡터로 변환함
- lexical level feature는 주어진 명사들(nouns)에 따라 추출됨
- sentence lebel feature는 CNN 학습과정 중에 추출됨
- 이렇게 추출한 두 feature를 softmax classifier를 통해서 마킹한 두 명사(entity) 간의 관계를 예측함
- state of the art보다 상당히 성능이 좋음

## 1. Introduction

- 대부분 supervised learning을 함
- supervised learning은 크게 feature-based method와 kernel-based method가 있음
- feature-based method는 bag-of-words model 같은 거고 kernel-based method는 dependency parse tree 같은 거임
- 이런 방법은 효과적이지만 이런 모델 자체의 영향력이 너무 크고 보통 이런 feature나 kernel은 기존에 존재하는 NLP system에 의해 유도됨 -> 외부 NLP system에 의존적이다는 의미인듯
- 우리는 Deep CNN을 사용할 것임

## 2. Related Work

- skip...

## 3. Methodology

![network](https://user-images.githubusercontent.com/15166794/36367647-f98a3236-1596-11e8-973c-f27d3c89c073.png)

### 3-1. The Neural Network Architecture

- neural network architecture를 사용
- Word Representation, Feature Extraction, Output, 3가지 component로 구성됨
- 우리 시스템은 복잡한 전처리가 필요하지 않고 input으로 그냥 2개의 명사가 마킹된 문장이 들어감
- 단어 토큰은 lookup table에 기반하여 vector로 임베딩 됨
- lexical feature와 sentence feature는 각각 추출되고 연결(concatenate)시켜 최종 feature vector를 완성함
- 이 최종 feature vector는 softmax classifier를 통해서 output(prediction)을 생성해냄

### 3-2. Word Representation

- 랜덤 초기화보다 학습된 word embedding vector를 사용하는 게 좋고 우리는 Turian et al.(2010)의 trained embedding을 사용함

### 3-3. Lexical Level Features

- 기존의 lexical level feature는 주로 명사 자체, entity 쌍, entity 사이의 시퀀스를 포함시켰고 이들은 NLP tool에 의존적이였음
- 우리는 이런 방법 대신에 워드 임베딩 기반의 feature를 사용하였음
- 마킹된 명사와 그 주변의 context 토큰(단어)들, 그리고 MVRNN(Socher et al. 2012)에서처럼 WordNet hypernym을 아래와 같이 단계적으로 사용함
  - L1 = Noun 1
  - L2 = Noun 2
  - L3 = Left and right tokens of Noun 1
  - L4 = Left and tight tokens of Noun 2
  - L5 = WordNet hypernyms of nouns

### 3-4. Sentence Level Features

- 3-2에서 사용한 word vector는 word similarity를 잘 표현하지만 문장에서 long distance feature나 sementic compositionality 면에서 부족함
- 그래서 우리는 sentence level feature를 자동으로 추출할 수 있는 max pooled convolutional neural network를 사용함
- Figure 2는 sentence level feature 추출을 위한 cnn 모델임. 이는 전체 아키텍처를 나타내는 Figure 1의 sentence level features 부분에 들어간다고 보면 됨
- 최종적으로 우리는 non-linear한 sentence level feature를 얻을 수 있음

### 3-4-1. Word Features

- Distributional hypothesis theory (Harris, 1954)에 따르면 같은 context에서 나타나는 단어들은 유사한 의미를 지니는 경향이 있음
- 이러한 특성을 잡기 위해 해당 단어와 주변 context 단어의 vector representation을 조합하기로 함
- *[People]\_0 have\_1 been\_2 moving\_3 back\_4 into\_5 [downtown]\_6* 와 같은 단어가 있다고 했을 때, 각 단어를 임베딩 시키면 (x0, x1, x2, ... , x5, x6)이 될 것임
- 이 벡터 리스트를 window_size = 3으로 주변 단어를 묶으면 ([x_start, x0, x1], [x0, x1, x2], ... , [x4, x5, x6], [x5, x6, x_end])을 얻을 수 있고 이를 WF(Word Features)로 사용함

### 3-4-2. Position Features

- 이전에 relation classification을 위해 structure features(e.g., the shortest dependency path between nominals)를 사용했음. 하지만 WF는 이런 feature를 포함하지 못함
- 우리는 Position Features(PF)를 추출하기 위해 상대적인 거리를 이용함. 예를 들어 위의 3-4-1의 문장에서 moving은 마킹된 단어들(people, downtown)에 대해 (3, -3)의 상대적인 거리를 갖음
- 상대적 거리 값에 대한 임베딩 벡터(size is hyperparameter) d1, d2를 얻을 수 있고 PF = [d1, d2]로 만들어낼 수 있음
- 이렇게 얻은 position features를 word features와 합쳐서 [WF, PF] 벡터를 convolution component로 전달시킴

## 4. Datasets and Evaluation metrics

- SemEval 2010 Task 8 dataset
- macro average F1 score
