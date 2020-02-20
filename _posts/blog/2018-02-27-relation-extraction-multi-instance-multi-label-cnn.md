---
title: "Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks (COLING 2016)"
layout: post
date: 2018-02-27
headerImage: false
tag:
- relation-extraction
- multi-instance
- multi-label
- convolutional-neural-network
- cnn
category: blog
author: roomylee
---

- Paper Link: <https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf>
- Author
  - Xiaotian Jiang (University of Chinese Academy of Sciences)
  - Quan Wang (University of Chinese Academy of Sciences)
  - Peng Li (University of Chinese Academy of Sciences)
  - Bin Wang (University of Chinese Academy of Sciences)
- Published at
  - COLING 2016

---

## Abstract

- Distant supervision은 relation extraction을 위해 자동으로 labeled data를 생성하는데 효과적인 방법임
- 전통적인 방식들은 handcrafted feature에 상당히 의존적이고 이에 대한 error가 그대로 전해지게 됨
- 그래서 최근엔 neural network를 이용해서 relation classification을 위해 자동으로 feature를 뽑아주는 방식이 제안되고 있음
- 하지만 이런 방식들은 traditional expressed-at-least-once assumption을 두고 있고, 여러 문장에 걸쳐서 나타나는 정보를 활용할 수 없음
- 또한, 동일한 entity pair에 대해서 여러 개의 relation이 나타날 수 있다는 사실을 무시하고 있음
- 이 논문에서 우리는 multi-instance multi-label cnn for distantly supervised RE를 제안하는 바임
- 우선 expressed-at-least-once assumption을 완화시킬 것이고, cross-sentence max-pooling을 통해 다른 문장 간의 정보 공유를 할 수 있도록 할 것임
- multi-label learning을 통해서 overlapping relation을 다룰 수 있음
- state-of-the-art 보다 상당히 좋음

## 1. Introduction

- Relation extraction은 binary relation을 plain text로 부터 추출하는 task라고 정의함
- Supervised method들이 높은 성능으로 인해 널리 사용됨
  - 하지만 human annotation이 필요하고 이를 만들기 위해 시간이 많이 소모된다는 단점이 있음
  - 그래서 knowledge base 기반으로 자동으로 labeled data를 생성해주는 distant supervision이라는 기법이 등장함
- supervised RE는 POS tags, dependency path, named entity tags 등의 lexical and syntactic feature가 사용됨
- 하지만 이런 feature들은 NLP algorithm(tool)을 사용하고 있고 이 때문에 error를 갖게 됨
- 이런 error는 긴 문장일수록 더 심각한 문제를 초래하는데, 불행하게도 이런 긴 문장들이 코퍼스의 대부분을 차지하고 있음
- 문제가 있는 feature를 사용하는 distant supervision 방법은 error를 전파시키고 성능 저하의 주범이 됨
- 그래서 최근에는 자동으로 feature를 추출하는 deep nearal network 기반의 연구가 많이 진행되고 있음
- 특히 piecewise convolutional neural network (PCNN)이 좋은 결과를 내고 distant supervised relation extraction에서 상당한 향상을 보였으나 아직 (아래의) 몇가지 결점을 가지고 있음

![figure1](https://user-images.githubusercontent.com/15166794/36642226-b7ebadd0-1a7f-11e8-8b01-a43aab6473d4.png)

- 첫째, PCNN은 labeled data를 생성해내는데 expressed-at-least-once assumption을 사용한다는 것임
  - expressed-at-least-once assumption이란, 두 entity가 어떤 relation을 갖고 있을 때, 어떤 문장에서 두 entity가 등장하면 이는 그 relation을 갖고 있다고 보는 것임
  - 그러나 이 가정은 너무나도 강력하고, 한 문장을 선택하는 것은 다른 문장들로부터 얻을 수 있는 많은 정보를 잃은다고 봄
  - 실제로 knowledge base relation에 나타난 두 entity가 주어졌을 때, 그 relation을 정확히 표현하는 하나의 문장을 training text로 부터 찾는 것은 상당히 어려움
  - Figure 1을 보면 *Thailand* 와 *Bangkok* 의 entity pair를 포함한 세 문장이 있음
  - 이 문장들 중 어느 것 하나 */location/country/capital* relation을 나타내지는 않음
  - 하지만 이 세 문장을 종합적으로 봤을 때는 */location/country/capital* relation을 유추해볼 수 있다는 것임

- 둘째, PCNN은 single-label learning problem으로 distantly supervised RE를 다루고 있음
  - 선택된 두 entity pair는 여러 개의 relation을 갖고 있더라도 반드시 하나의 relation label을 갖는다고 봄
  - New York Times 2007 corpus의 약 18%가 overlapping relation을 갖는 문장인 것으로 나타남
  - 따라서 single-label learning은 문제가 있음
- 이 논문에서 우리는 위의 문제들을 해결하기 위해 multi-instance multi-label convolutional neural network (MIMLCNN)를 제안함
- 첫 번째 문제에 대해서는 기존의 expressed-at-least-once assumption 대신에 *"a relation holding between two entities can be either expressed explicitly or inferred implicitly from all sentences that mention these two entities" -> 두 entity의 relation은 분명하게 표현될 수도 있고 (이게 기존 가정인듯), 언급된 모든 문장들로부터 함축적으로 추론될 수도 있다 (Figure 1의 내용)* 라는 보다 완화된 가정을 둘 것임
  - 구체적으로는 아래의 순서대로 하면 됨
  - 각 문장 별로 convolution을 시켜서 feature를 추출함
  - 새로 제안하는 cross-sentence max-pooling이라는 것을 통해 다른 문장에 걸쳐 나타나는 feature를 추출함
  - 그리고 most significant feature(max?)를 합쳐서 각 entity pair의 vector representation으로 만듬
  - 이 vector representation은 다른 문장들의 feature를 포함하기 때문에 가정에서 말했던 여러 문장들의 정보를 모두 사용하는 셈이 됨

- 두 번째 문제에 대해서는 다양한 multi-label loss function을 만들어서 overlapping relation을 처리할 수 잇도록 함
  - 아래의 section 3에 나오는 Figure 2를 참고하면 전체적인 구조를 알 수 있음
- 우리 논문의 메인 contribution은 아래와 같음
  1. expressed-at-least-once assumption을 완화시켰고 여러 문장들이 서로 정보를 공유할 수 있게 하는 더욱 현실적인 가정을 제안함
  2. multi-label을 다룰 수 있는 multi-instance multi-label convolutinoal neural nerwork (MIMLCNN)을 제안함
  3. 우리는 real-world dataset으로 우리의 approach를 평가했으며, state-of-the-art 보다 상당한 향상을 보였음

## 2. Related Work

- skip...

## 3. Our Approach

- 공통의 entity pair(e1, e1)를 갖는 sentence가 input으로 들어가고, knowledge base에서 정의된 relation(class)가 output으로 나오게 됨. Figure 2 참고
- 우리의 approach는 3가지 key step을 갖는데, (1) sentence-level feature extraction, (2) cross-sentence max-pooling, (3) multi-label relation modeling 임. Figure 2 참고

![figure2](https://user-images.githubusercontent.com/15166794/36655746-ebb035cc-1b07-11e8-87fb-798eee7f5a22.png)

### 3.1 Sentence-level Feature Extraction

- sentence-level feature extraction은 문장으로부터 vector feature를 만들어내는 단계임
- 기존에 제안되었던 text cnn, cnn for RE, piecewise max pooling 등을 그대로 사용하고 있음
- 최종적으로 sentence-level feature extraction 결과를 **sentence representation (vector)**이라고 함
- Figure 3 참고

![figure3](https://user-images.githubusercontent.com/15166794/36655747-ebd89f6c-1b07-11e8-9246-5d6617c33a52.png)

### 3.2 Cross-sentence Max-pooling

- 두 entity의 relation을 한 문장이 아닌 여러 개의 문장의 정보를 이용해서 예측하겠다는 것이 이 논문에서 주장하는 바임
- 그래서 아래와 같은 가정을 둠. 이는 이전 PCNN에서 사용한 가정보다 완화된 것임
- Assumption: *A relation holding between two entities can be either expressed explicitly or inferred implicitly from all sentences that mention these two entities.*
  - 이 가정의 본질에 의하면, 우리는 sentence-level relation extraction을 건너 뛰고 entity-pair-level에서 prediction을 직접적으로 하게 됨
  - 이는 더욱 downstream apllication을 염려하고 더 근거를 통합하는데 이득임
  - 뭔 소린지를 모르겠음
- 우리는 이 가정의 장점을 가지는 cross-sentence max-pooling이라는 방법을 제안하는 바임
  - entity pair가 언급된 m개의 문장이 있다고 하자
  - 각 문장은 앞의 sentence-level feature extraction 과정을 거쳐서 sentence representation (vector)로 변환됨
  - 즉, sentence representation vector가 m개가 생기게 되는 것이고, 이 m개의 벡터들의 각 원소에 대해서 최대값을 추출하여 하나의 벡터로 pooling하는 것이 **cross-sentence max-pooling**임. 각 원소 별로 m개 중에 최대를 뽑아내는 것임
- 이 작업을 하면 몇 가지 이익이 있음

 1. 각 문장으로부터 feature를 통합하게 되고 entity-pair-level relation extraction을 지원(support)함
 2. 다른 문장들로부터 relation 예측에 대한 근거를 모을 수 있음
 3. Zeng et al. (2015)이 하나의 문장씩 학습한 것에 비해 우리는 가능한 모든 문장으로부터의 정보를 모두 사용하여 학습한다는 장점이 있음

- mean-pooling 같은 방법을 쓰지 않고 max-pooling을 사용한 이유는 다음과 같음
  - entity-pair-level relation extraction에 있어서 여러번 등장한 feature는 더 많은 추가 정보를 주지 않는다고 생각함
  - 즉, 한 번씩만 나타난 구분 가능한(서로 다른) signal만으로도 relation extraction에 있어서 충분함
  - 각 feature의 maximum activation level만을 여러 문장에 걸쳐서 하나씩 뽑는 cross-sentence max-pooling 방법이 바로 이런 생각을 구현한 것임
  - 반면, mean-pooling 같은 경우에는 여러번 언급된 entity-pair에 대해서 predictive feature가 희석될 수 있음
  - 위의 주장은 뒤에 실험을 보면 더욱 잘 알 수 있음

### 3.3 Multi-label Relation Modeling

- 기존의 multi-instance learning을 적용한 neural network 방법들은 어떤 entity pair가 여러 개의 relation을 갖고 있어도 single label로 학습함
- 우선 pooling을 거쳐서 나온 벡터에 FC를 붙이고 그 결과값에 sigmoid를 취함
- binary label vector인 y는 relation이 있으면 1, 없으면 0으로 표기 되고 복수의 relation이 있으면 1이 여러 개인 vector가 되는 것임
- 이 방법으로 하면 아무 relation이 없는 NA 케이스에 대해서 all-zero vector를 사용하면 되기에 표현이 자연스러워짐. 기존에는 NA 클래스에 해당하는 index 하나를 더 만들어서 one-hot vector로 구성함
- relation 간에 dependency가 걸리는 경우도 있음
- (A, capital, B)와 (A, contains, B)는 거의 붙어다닐 것인데, 우리의 모델은 모든 relation label에 대해서 shared entity-pair-level representation을 사용하므로 처리할 수 있음
- multi-label modeling을 위해서 아래의 두 loss function을 만듬

![formula](https://user-images.githubusercontent.com/15166794/36669255-9866812a-1b37-11e8-9d3a-84c042316502.png)

- 위의 두 loss function을 사용하고 이에 대한 실험을 뒤에 할 것임
- 우리의 모델은 end-to-end로 학습하고 Adadelta를 optimizer로 사용하며 dropout이 적용되어 있음
- 최종적으로 prediction vector에서 각 원소 값(확률)이 0.5 초과할 경우, 해당 label을 1로 처리하여 output을 내보냄

## 4. Experiments

- NYT10을 이용해서 평가를 진행
- precision-recall curve와 P@N metric(확실한 순으로 상위 N개에 대한 precision)을 사용함

## 5. Conclusion

- 우리는 Distant supervision with multi-instance multi-label learning을 제안함
- expressed-at-least-once assumption을 완화시키고, 여러 문장에 걸쳐 나타나는 정보를 cross-sentence max-pooling을 통해 모두 사용할 수 있도록 하였으며, multiple relation을 갖는 entity pair에 대한 modeling을 해봄
- future work로는 loss function이 성능에 미치는 영향에 대한 분석이나 human evaluation을 해보면서 실험을 보다 풍부하게 만들어볼 수 있을 것 등이 있음
