---
title: "Incorporating Relation Paths in Neural Relation Extraction"
layout: post
date: 2018-02-27
headerImage: false
tag:
- relation-path
- neural-relation-extraction
- relation-extraction
category: blog
author: roomylee
---

- Paper Link: <#  [[pdf]](http://aclweb.org/anthology/D17-1186)
- Author
  - Wenyuan Zeng (Tsinghua University)
  - Yankai Lin (Tsinghua University)
  - Zhiyuan Liu (Tsinghua University)
  - Maosong Sun (Tsinghua University)
- Published at
  - EMNLP 2017

---

## Abstract

- Distantly supervised relation extraction은 plain text로부터 새로운 relation fact를 찾아내기 위해 널리 사용됨
- Target entity pair 사이의 relation을 예측하기 위한 현재의 방법들은 entity pair가 포함된 문장 그 자체에 대해 의존적임
- 사실, target entity pair 중 하나의 entity만을 포함한 문장은 매우 많이 있고 이 문장들은 매우 유용한 정보를 제공하지만, 아직 relation extraction에 이용된 바가 없음
- 이런 이슈를 처리하기 위해,
  - 우리는 인접한 entity들을 통해서 두 target entity 간의 추론 연쇄(inference chains)를 만듬
  - 그리고 문장 자체와 inference chains으로부터 relational semantics를 encoding하기 위한 path-based neural relation extraction model를 제안하는 바임
- real-world datasets으로 실험한 결과를 통해서,
  - 우리의 모델이 하나의 target entity만 나타난 문장들을 완전히 사용하였고
  - relation extraction의 baseline들에 비해 상당한 성능 향상을 가져왔다는 것을 보여줄 것임
- Github에 소스를 공개함. [https://github.com/thunlp/PathNRE](https://github.com/thunlp/PathNRE)

## 1. Introduction

- Knowledge Bases (KBs)는 현실의 사실들에 대한 효과적인 정형 데이터를 제공하고 Web search나 QA 등의 NLP 응용 분야에서 중요한 자원으로 사용됨
- Freebase, DBpedia, YAGO 등의 KB를 많이 사용하는데 이들은 multi-relational data이고 triple 형태로 표현됨
- Real world의 fact는 무한이 증가하는데 비해, 현존하는 KB는 이에 비해 한참 부족함
- 최근에는 많은 양의 다른 structure type을 포함하는 petabytes 단위의 자연어 text를 이용할 수 있고, 이는 자동으로 모르는(unknown) relational fact를 찾는데 중요한 자원이 됨
- 이때문에 RE는 plain text로부터 정형 정보(structured information)를 추출하는 테스크로 정의함
- 대부분의 supervised RE system들은 labeled data가 충분하지 못함. 수동 태깅은 시간과 노동력이 많이 듬
- 이를 해결하기 위해 KB를 이용해 plain text로부터 자동으로 training data를 만들어주는 distant supervision이 나옴. 그리고 여기에 더해서 neural model을 만드는 노력들이 있었음
- 이런 model들에는 한 가지 중요한 결점이 있는데, 바로 두 target entity가 나타나는 문장만으로 학습을 한다는 것임
- **그러나 오직 하나의 entity만 나타나는 문장도 유용한 정보를 제공하고 inference chains을 만드는데 도움이 됨**
- **예를 들어, "*h* is father of *e*" and "*e* is the father of *t*" 라는 두 문장이 있으면 우리는 *h*가 *t*의 할아버지(grandfather)라는 사실을 유추할 수 있다는 거임**

![figure1](https://user-images.githubusercontent.com/15166794/36712183-e6782d6c-1bca-11e8-8a8d-6c8d4ff5d1f9.png)

- 이 연구에서 우리는 Figure 1과 같은 relation path를 통한 path-based neural relation extraction model을 제안하는 바임
  - 첫째, 우리는 CNN을 이용해서 문장의 의미(semantics)를 임베딩시키고자 함
  - 그러고나서, 우리는 inference chain이 주어졌을 때(given) 각 relation들의 확률을 측정할 수 있는 relation path encoder를 만듬
  - 마지막으로 relation을 prediction하기 위해 위의 두 정보인 direct sentences와 relation path를 합침(combination)
- Real-world dataset으로 평가했고 baseline들보다 상당히 좋은 성능을 보임
- entity 하나만 등장하는 문장도 이용하므로서 우리의 모델은 더욱 robust하고 noisy instance가 증가해도 잘 작동함
- Plain text에서 relation path를 이용해서 neural relation extraction을 한 최초의 연구임

## 2. Related Work

- skip...

## 3. Our Method

- [target entity pair, entity pair를 포함한 문장들, relation path들], 총 3가지가 주어졌을 때, 우리의 모델은 해당 entity pair에 대한 각 relation의 신뢰도(confidence)를 측정하는 역할을 함
- 이번 section에서 우리는 모델의 3가지 파트를 소개할 거임
  1. **Text Encoder**: target entity pair를 포함한 문장이 주어지면, CNN으로 문장을 semantic space에 임베딩시키고 각 relation에 해당될 확률을 구함
  2. **Relation Path Encoder**: 두 target entity 사이의 relation path가 주어지면, 이 path를 조건으로 하여 각 relation에 해당될 확률을 구함
  3. **Joint Model**: 1. direct sentences 와 2. relation paths 의 정보를 통합하여 각 relation class에 대한 confidence를 예측함

![figure2](https://user-images.githubusercontent.com/15166794/36712296-7e91bfd2-1bcb-11e8-951d-bf64868ad400.png)

### 3.1. Text Encoder

- 기본적인 CNN for RE 모델
- word and position embeddings -> text convolution -> max pooling -> tanh -> FC -> softmax
- output에 대해서 multi-instance learning을 진행
- multi-instance learning이란 각 relation에 대해서 가장 확률이 높은(가장 relation이 명확한) 문장만 뽑아서 학습하는 방식임. 이전 기저(basis) 논문에서는 bag이라는 걸 뒀던 거 같은데 그런 언급은 없음

### 3.2. Relation Path Encoder

- 우리는 relation path의 추론 정보를 embedding하기 위해서 Relation Path Encoder를 사용함
- Relation Path Encoder는 relation path가 주어졌을 때 (given), 각 relation에 해당될 확률을 측정함

![eq](https://user-images.githubusercontent.com/15166794/36721730-170dc6ca-1bef-11e8-912a-84e24f2fcfe3.png)

- 예를 들어,
  - (h, t) pair 간의 path p1을 {(h, e), (e, t)}라고 하고 이는 rA, rB에 해당된다고 하자
  - (h, e)와 (e, t) 각각은 최소 하나 이상의 sentence에서 나타난다고(correspond) 하자
  - 이때 우리의 모델은 위의 Equation 8의 식으로 확률을 계산함
  - Equation 8의 설명을 위해 Equation 9를 먼저 보자
    - Equation 9는 특정 relaion과, rA와 rB를 연쇄시킨 relation과의 유사도를 의미함
    - 연쇄시킨 relation은 실제하는 클래스가 아니고 rA와 rB의 합임
    - rA와 rB를 비롯한 모든 relation class는 binary vector representation이 아닌 distributed representation이고 따라서 위에서 말한 rA와 rB의 합은 벡터 element-wise sum이라고 볼 수 있음
    - (-)를 붙인 이유는 L1(절대값)이 작을수록(0에 가까울수록) ri와 (rA+rB)가 유사한 것을 표현하기 위해서임
  - 따라서 Equation 8은 이런 유사도(Equation 9)에 대해서 softmax를 취해서 확률값으로 변환시킨 셈이 됨
  - 그렇기에 Equation 8에서 p(r | rA,rB)의 의미와 우변의 식은 일맥상통한다고 볼 수 있는 것임
  - 앞에 Equation 9에서 (-)를 붙였기에 유사도가 높을수록(Equation 9가 작을수록) p(r | rA,rB)가 커지게 됨
- 위의 예시에서, Equation 9의 ri가 relation path pi: (h --rA-> e --rB-> t) 의미적으로 유사하다면 ri의 embedding 값이랑 (rA+rB)의 임베딩 값이 매우 가깝다는 것을 가정으로 함
- 즉, rA=father, rB=mother 라고 하면 father vector + mother vector = spouse vector가 되도록 relation class vector embeddings을 학습시키는 느낌이지 않을까 싶음

![eq2](https://user-images.githubusercontent.com/15166794/36728622-b7bea584-1c04-11e8-9026-c617d3d98846.png)

- 결과적으로 relation-path score function은 Equation 10과 같이 정의됨
- E(h,rA,e)는 entity h와 e가 등장하는 문장에서 두 entity가 rA라는 관계를 가질 확률을 의미함
- 실제로는 두 entity 사이에 여러 개의 relation path가 존재하기 때문에 Equation 11을 통해 최종적으로 가장 확률 높은 relation path를 얻게 됨

### 3.3. Joint Model

![eq3](https://user-images.githubusercontent.com/15166794/36729969-36f87c9a-1c09-11e8-845a-dc6c720d885f.png)

- Direct sentence에 대한 score와 relation-path score를 합친(joint) 최종 global score function은 위의 Equation 12와 같음
- 여기서 α = (1 - E(h,r,t | S)) * β 임. β는 상수이며 direct sentence와 relation path 간의 상대적 비중을 조정하는 인자임
- α를 E(h,r,t | S)에 대한 식으로 둔 이유는, 만약 E(h,r,t | S)가 충분히 크다면 굳이 extra 정보인 relation path에 주목하지 않아도 충분히 신뢰할만한 prediction을 할 수 있기 때문임
- Joint model의 장점 중 하나는 error propagation을 완화시킬 수 있다는 것임
- 그냥 direct sentence와 relation-path 간에 불확실성을 서로 적절히 보완해준다고 말하는 것 같음. (당연한 얘긴데 왜 장점이라고 굳이 집어서 말하는지는 모르겠음)

### 3.4. Optimization and Implementation Details

![eq4](https://user-images.githubusercontent.com/15166794/36730716-c583fa96-1c0b-11e8-8142-938365e6b185.png)

- 위의 식(Equation 13)이 바로 objective function (loss function)임
- Stochastic gradient descent를 사용해서 objective function을 최대화하는 방향으로 학습(optimization)함
- W_E만 skip-gram model로 초기화하고 나머지 파라미터는 다 랜덤
- output layer에 dropout 적용
- relation path structure는 학습 이전에 추출하고 저장해뒀음. 추출을 어떻게 했는지는 언급 없음

## 4. Dataset

- New York Times corpus
- ACE는 데이터가 너무 적어서 학습시키기 부적합하다고 봄

## 5. Experiments

- Precision/Recall curve와 Precision@N (P@N)과 F1 score를 지표로 사용함
- 이전 Zeng et al.의 2014년 논문의 CNN 모델과 2015년 논문의 CNN(PCNN, multi-instance learning), 두 가지를 비교 모델로 사용함

## 6. Conclusion and Future Work

- Relation path를 encoding 시킨 neural relation extraction을 해봄
- Entity pair 중 두 개 모두 혹은 한 개의 entity만이라도 포함된 문장을 이용하였고 덕분에 noisy data에 대해 더욱 robust해짐
- 이전 baseline에 비해 상당한 성능 향상
- Future work으로는, (1) platin text와 KB의 relation path를 조합해서 사용해보기, (2) rnn 같은 거로 relation path 간의 더욱 복잡한 상관성을 encoding하기 (우리는 2 step path였지만, multi-step path를 이용)
