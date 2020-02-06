---
title: "Relation Extraction: Perspective from Convolutional Neural Networks"
layout: post
date: 2018-02-19
headerImage: false
tag:
- relation-extraction
- cnn
- analysis
category: blog
author: roomylee
---

- Paper Link: <http://www.cs.nyu.edu/~thien/pubs/vector15.pdf>
- Author
  - Thien Huu Nguyen (New York University)
  - Ralph Grishman (New York University)
- Published at
  - NAACL 2015

---

## Abstract

- In relation extraction, tranditional approach with complicated feature engineering has errors and it lead to errors of relation detection and classification.
  - ➤ 관계추출에서 전통적인 방법은 복잡한 feature engineering을 하기에 에러가 많고 때문에 detection과 classification에서 많은 문제를 야기한다.
- Advantage of our model: multiple window sizes for filters.
  - ➤ 필터의 윈도우 사이즈를 여러가지로 할 수 있다.
- using pre-trained word embeddings as initializer.
  - ➤ pre-trained 된 워드 임베딩 모델을 사용하고 있다.

## 1. Introduction

- Relation Extaction task can be divided into two steps: 1) Detecting and 2) Classifying
  - ➤ 관계추출 문제는 관계를 찾아내는 단계와 이를 분류하는 단계, 두 가지로 나눌 수 있다.
- Difference between Relation Classification and Relation Extraction ➤ RC와 RE의 차이점
  - In classification, non-relation examples in the dataset are comparable to the other examples, so they can be treated as a usual relation class like *Other- class.(balanced)
  - ➤ RC에서는 non-relation인 데이터의 양이 relation인 데이터 양과 비슷하다.(balanced) 그래서 보통의 클래스처럼 이를 다룰 수 있다.
  - In Extraction, non-relation examples far exceeds the others.(unbalanced) So more challenging but more practical than relation classification.
  - ➤ RE에서는 non-relation인 데이터의 양이 매우 많다.(unbalanced) 그래서 더 어렵고 쓸모가 있다.
- In the last decade, the relation extraction has been dominated by two methods, the feature-based method and kernel-based method.
  - ➤ 지금까지 relation extraction의 해법으로 feature-based와 kernel-based 두 가지 방법이 지배적이었다.
- The common characteristic of these methods is the leverage of a large body of linguistic analysis and knowledge resourses to transform relation mentions into some rich representation to be used by some statistical classifier such as SVM, MaxEnt.
  - ➤ 두 방법의 공통적인 특징은 relation mention을 통계적 분류기(SVM, MaxEnt)에서 사용할 수 있는 풍부한 표현으로 변환하기 위해서 *언어 분석 및 지식 자원의 많은 부분을 활용*한다는 것이다.
- So these models depend on a supervised NLP toolkit and suffer from a performance loss when they are applied to out-of-domain data.
  - ➤ 그래서 이 모델들은 학습된 NLP toolkit에 의존적이고 out-of-domain 데이터에 대해서 퍼포먼스 소실이 있다.
- We target an independent RE system that both avoids complicated feature engineering and minimizes the reliance on the supervised NLP modules.
  - ➤ 우리는 복잡한 feature engineering과 NLP module의 의존성을 최소화한 독립적인 RE 시스템을 목표로 하였다.
- there are two recent works on CNNs for relation classification (Liu et al., 2013) and (Zeng et al., 2014); however work on CNNs for relation extraction is not yet.
  - ➤ CNN을 통한 RC에 대해서는 최근 Liu와 Zeng의 연구가 발표된 바 있지만, RE는 아직 발표된 게 없다.

## 2. Related Work

- skip...

## 3. Convolutional Neural Network for Relation Extraction

![network](https://user-images.githubusercontent.com/15166794/36367668-1a3dbc00-1597-11e8-8a8c-74b6607567f6.png)

- There are main 4 layers
  1. the lookup tables to encode words in sentences by real-valued vectors
  2. the convolutional layer to recognize n-gram
  3. the pooling layer to determine the most relevant features
  4. the logistic regression layer (FC with softmax at the end) to perform classification

### 3-1. Word Representation

- CNN은 fixed length input이어야만 함
- fixed length는 relation을 갖는 두 entity 사이의 최대 거리로 하였고, 이 length보다 긴 문장은 자르고 짧은 문장은 special token으로 padding 시킴
- 문장을 이루는 각 단어 토큰은 임베딩 lookup table(random or pre-trained)에 기반하여 벡터로 변환시킴
- 마킹된 두 entity의 position을 임베딩하기위해 각 entity와 모든 단어에 대한 상대적 거리 값(i-i1, i-i2)을 구함
- 이를 가지고 random initialize된 real-value vector(d1, d2)로 변환할 수 있는 lookup table을 만듬
- 따라서 상대적 거리 값의 범위는 -n+1부터 n-1까지이고, position embedding lookup table은 (2n-1) - m_d의 사이즈를 갖고 이때 m_d는 hyperparameter인 position embedding size를 의미함
- 최종적으로 문장의 각 단어 토큰은 word embedding vector, position embedding vector d1과 d2를 이어 붙인(concatenate) vector를 갖게 됨
- 따라서 하나의 단어 토큰은 (word embedding size + 2*position embedding size)의 차원을 갖는 벡터가 됨

### 3-2 & 3. Convolution & Pooling

- 다양한 size의 filter를 이용해서 단어 시퀀스(문장)을 convolution 시키며, bias와 non-linear activation function이 사용됨
- convolution된 값(벡터)에 대해 max pooling을 진행하고 max pooling할 벡터의 크기는 convolution한 filter의 크기에 따라서 (fixed length - filter size +  1)의 길이를 갖음

### 3-4. Regularization and Classification

- dropout
- FC
- softmax
- l2 norm
- AdaDelta

## 4. Experiments

- Dataset
  - SemEval 2010 Task 8 dataset for relation classification
  - ACE 2005 dataset for relation extraction
- static & non-static
  - static은 word & position embedding vector를 학습하지 않는 것, non은 그 반대
  - random init + non-static, pre-trained + non-static, pre-trained + static, 3가지에 대해서 성능 평가를 해봄
- tanh for non-linear activation function
- 150 filters for each window size
- word embedding size = 300 (using GoogleNews word2vec)
- position embedding size = 50^4
- dropout keep prob = 0.5
- batch size = 50
- hyperparameter of l2 = 3

## 5. Conclusion

- 우리는 CNN으로 unbalanced corpus에 대해서도 잘 작동하고 feature를 위한 외부 supervised NLP toolkit의 사용을 최소화했음
- 아래의 측면에서 relation classification & extraction task에 대해 기여(contribution)가 있다고 봄
  - multiple window size
  - position embedding
  - pre-trained word embedding for init in a non-static architecture
- future work로는 relation extraction을 위한 CNN의 추가적인 feature 추출 방법 고안, CNN말고 다른 뉴럴넷을 이용한 relation extraction 문제 해결 등이 있다고 봄
