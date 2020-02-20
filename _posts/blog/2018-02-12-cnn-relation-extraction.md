---
title: "Convolution Neural Network for Relation Extraction (ADMA 2013)"
layout: post
date: 2018-02-12
headerImage: false
tag:
- convolutional-neural-network
- cnn
- relation-extraction
category: blog
author: roomylee
---

- Paper Link: <https://link.springer.com/chapter/10.1007/978-3-642-53917-6_21>
- Author
  - ChunYang Liu (National Computer Network Center of China)
  - WenBo Sun (Beijing University)
  - WenHan Chao (Beijing University)
  - WanXiang Che (Harbin Institute of Technology)
- Published at
  - ADMA 2013

---

## Abstract

- CNN을 사용함
- synonym dictionary를 사용하여 new coding(embedding) 방법을 제안
- 기존 방식(tree kernel 기반)보다 약 9%정도 좋음
- ACE 2005 dataset을 사용
- hypernym에 대해서도 실험을 해봄

## 1. Introduction

- 커널 기반 모델은 파싱을 해야하고 이는 복잡도가 매우 큼
- 또한, 수동으로 사람이 직접 feature engineering을 해야함
- 이전 연구에서는 임베딩 방법이 semantic 의미를 담지 못하였으나 우리는 그걸 담고 있는 synonym coding이라는 방법을 사용함

## 2. Related Work

- skip...

## 3. Convolution Network Architecture

- one-hot 인코딩 방법 대신에 synonym dictionary에 기반한 임베딩 방법을 사용
- 해당 synonym임베딩 인덱스 벡터가 input으로 들어감
- 그러고 나서 lookup table layer를 통해 벡터 변환
- CNN -> FC -> softmax 를 거쳐 아웃풋을 뽑음

## 4. Experiments

- 성능은 상당히 잘 나옴
- 여러가지 실험을 했는데 정확히 어떤 데이터인지 불명확하고 다른 논문으로 보아 왠지 실험 결과가 의심스러움
