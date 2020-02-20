---
title: "Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks (EMNLP 2015)"
layout: post
date: 2018-02-20
headerImage: false
tag:
- piecewise-cnn
- relatin-extraction
- distant-supervision
category: blog
author: roomylee
---

- Paper Link: <http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf>
- Author
  - Daojian Zeng (Chinese Academy of Sciences)
  - Kang Liu (Chinese Academy of Sciences)
  - Yubo Chen (Chinese Academy of Sciences)
  - Jun Zhao (Chinese Academy of Sciences)
- Published at
  - EMNLP 2015

---

## Abstract

- Distant supervision에는 2가지 문제점이 있음
  - 첫 번째 문제점은 distant supervision에서 이미 존재하는 knowledge base를 이용하는데 있음
  - knowledge base는 휴리스틱하게 만들어지고 이를 이용해서 labeled data를 만들어서 학습을 하는데, 휴리스틱은 실패할 수 있고 그렇게 만든 knowledge base와 그 결과물인 labeled data 역시 문제(wrong label problem)가 있을 수 있다는 것임
  - 두 번째는 이전 연구의 확률적 모델은 대개 ad hoc feature를 이용하는데 이 feature를 추출하는 과정 자체에 noise가 있고 이 noise가 성능 저하를 가져온다는 것임
- 우리는 위의 2가지 문제점을 다루기 위해, multi-instance learning을 이용한 Piecewise Convolutional Neural Networks (PCNNs) 라는 새로운 모델을 제안하는 거임
  - 첫 번째 문제를 해결하기 위해, distant supervised relation extraction을 label의 불확실성을 고려한 multi-instance 문제로 보고 처리함
  - 두 번째 문제를 해결하기 위해서는 feature engineering을 하지 않는 대신, piecewise max pooling을 적용한 convolutional architecture를 사용해서 관련있는 feature를 자동으로 학습시킴
- 몇몇 다른 방법들보다 효율적이고 좋은 성능을 보임

## 1. Introduction

- Relation extraction에서 하나의 도전 과제는 training examples을 만드는 것임

![figure1](https://user-images.githubusercontent.com/15166794/36376357-2a9912f6-15b6-11e8-9830-b8e9c1b57033.png)

- Distant supervision이 이를 해결하는 하나의 방법이 될 수 있음
- Distant supervision은 어떤 두 entity가 이미 알려진 knowledge base에서 relation을 가지고 있을 때, 이 두 entity가 등장한 모든 문장에 대해서 동일한 relation을 가진다고 가정하고 data를 만들어내는 것임 (Figure 1 참고)
- 하지만 이 방법은 2가지 중요한 결점을 가지고 있음
- 첫 번째는, distant supervision의 가정이 너무 강력하고 잘못된 label을 만들어낸다는 것임
  - 즉, 두 entity가 언급된 문장이 반드시 knowledge base에서 나타내는 relation을 포함한다고 볼 수 없다는 것임
  - Figure 1의 2 번째 문장에서 두 entitiy는 "/copany/founders"의 relation을 갖는다고 보기 어렵고 이런 noisy data 때문에 성능 저하가 일어나는 것임
- 두 번째는, distant supervision으로 데이터를 얻을 때 정교하게 디자인 된 feature를 가지고 model에 적용시킨다는 것임
  - 문제의 대표적 원인은 이미 존재하는 NLP tool을 사용하는데 있음
  - 불가피하지만 NLP tool을 사용하면서 그에 내제된 error가 feature에 전달이 되는 셈임
  - 심지어는 문장의 길이가 길어질수록 더더욱 성능이 떨어지기에 단순히 에러가 존재하는 것이 아니고 점점 더 심각해진다고 봐야 함. 왜냐하면 성능이 떨어지는 긴 문장이 코퍼스의 절반 정도를 차지하기 때문임
- 위의 두 문제점을 해결하기 위해 multi-instance learning을 이용한 PCNNs 모델을 제안하는 바임
- 첫 번째 문제(wrong label)를 multi-instance learning으로 해결할 거임
- multi-instance learning
  - training set을 많은 bag으로 구성하고 각 bag에는 많은 instance가 들어있다고 봄
  - bag의 label은 알고 있지만(known), bag를 구성하는 각 instance의 label은 모름(unknown)
  - 우리는 bag-level의 objective function을 학습시킬 것이고, 이렇게 하면 instance label의 불확실성을 고려할 수 있고 이는 wrong label problem을 완화시킬 수 있음
- 두 번째 문제(NLP tool)는 복잡한 NLP 전처리 과정을 없이 convolutional architecture가 자동으로 관련된 feature를 학습하도록 하여 해결함. 이는 Zeng et al. (2014)의 논문을 보면 됨
- 우리의 제안은 single max pooling을 사용한 Zeng et al.의 제안에 대한 확장판이라고 보면 됨
- 우리는 single max pooling이 아니라, 문장 구조적인 feature도 잡아내기 위해 주어진 두 entity에 의해서 만들어지는 3개의 segment에 대해 piecewise max pooling이라는 것을 할 것임
- 우리 paper의 contribution을 요약하면 아래와 같음
  - hand-designed feature 없이 distant supervised relation extraction을 위한 방법을 탐구하였고, 복잡한 NLP 전처리 없이 feature를 학습할 수 있는 PCNNs을 제안함
  - Wrong label problem을 해결하기 위해 multi-instance learning으로 PCNNs을 학습시켜 distant supervised relation extraction을 하는 해결책을 제시함
  - 문장에서 두 entity 간의 구조적 정보(structure information)를 잡아내기 위해, piecewise max pooling이라는 방법을 제안함

## 2. Related Work

- skip...

## 3. Methodology

- *3.1 Vector Representation*, *3.2 Convolution*, *3.4 Softmax Output* 은 skip...

![network](https://user-images.githubusercontent.com/15166794/36379736-111bd004-15c3-11e8-9b8b-9424dabb2d86.png)

### 3.3 Piecewise Max Pooling

- single max pooling은 relation extraction을 하기에 불충분함
- hidden layer의 size가 급격하고 거칠게 줄어들어서 고운 feature를 얻기 어려움 <br>➤ 전체 문장을 한순간에 하나의 값으로 뭉뚱그리기 때문에 feature가 뭉개진다는 의미인듯
- 그리고 두 entity에 대한 structural information을 잡아내기 어렵다는 단점이 있음
- piecewise max pooling은 위의 단점을 보완하는 방법으로서, 두 entity를 기준으로 문장을 3개의 segment로 나눈 뒤 각 segment 별로 max pooling을 해서 3차원의 벡터를 얻을 수 있음
- 이렇게 추출된 3차원 벡터들을 쭉 이어 붙이고(concatenate) non-linear activation function을 거쳐 다음 layer인 softmax output layer로 값을 보냄. 여기서는 tanh를 사용함

### 3.5 Multi-instance Learning

- wrong label problem을 해결하기 위해서 multi-instance learning for PCNNs을 사용함
- 아래의 설명을 틀릴 수 있음.
- bag라는 개념이 등장하는데, 이는 instance들의 집합이라고 보면 됨
- bag는 총 T개가 있고 target label의 개수와 같은 것으로 보임
- training step, 하나의 batch에서 모든 instance는 T개의 bag에 random하게 할당이 됨
- batch를 구성하는 모든 instance들은 network를 거쳐서 output(probability by softmax)을 구함
- 각 i번째 bag에서 i번째 label의 output(확률)이 가장 큰 instance를 해당 bag의 대표값으로 함
- 모든 bag의 대표값(output=확률)에 대해 cross-entropy를 구해서 네트워크 parameter를 업데이트 함
- 이렇게 하면 확실한 instance와 relation에 대해서만 학습을 진행하게 되어 wrong label problem을 완화시킬 수 있다고 함

## 4. Experiments

- Dataset
  - Riedel et al. (2010) 에서 만든 것을 사용함
  - Freebase relation(NYT corpus 기반)으로 만들어진 dataset임
  - 2005-2006을 training data, 2007을 testing data로 사용함
- Evaluation Metrics
- Mintz et al. (2009) 에서 사용한 held-out evaluation과 manual evaluation을 사용함
- 실험에 대한 precision/recall curve 등을 봄
- 나머지는 skip...

## 5. Conclusion
