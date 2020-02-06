---
title: "Recurrent Convolutional Neural Networks for Text Classification"
layout: post
date: 2018-04-21
headerImage: false
tag:
- text-classification
- recurrent-convolutional-neural-networks
- rcnn
category: blog
author: roomylee
---

- Paper Link: <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745>
- Author
  - Siwei Lai (Chinese Academy of Sciences)
  - Liheng Xu (Chinese Academy of Sciences)
  - Kang Liu (Chinese Academy of Sciences)
  - Jun Zhao (Chinese Academy of Sciences)
- Published at
  - AAAI 2015

---

## Abstract

- Traditional text classifier는 사전, knowledge base, tree kernel 등의 human-designed feature에 의존적임
- 우리는 이런 것에 의존적이지 않은 text classification을 위한 recurrent convolutional neural network을 제안함
- recurrent structure가 word representation을 학습할 때, 문맥적인 정보를 최대한 잡아냄
- 이는 traditional window-based neural network에 비해서 noise가 상당히 적음
- 우리는 또한 max-pooling layer를 두어서 자동으로 단어가 text classifiaction하는데 있어서 중요한 역할을 하는지 판단할 수 있도록 하였음
- 4개의 commonly used dataset으로 실험함
- 실험 결과는 몇몇 데이터에 대해서 SOTA를 능가하며, 특히 document-level dataset에서 더욱 그러함

## Introduction

- Text classification에서는 feature representation이 가장 핵심인데 전통적으로 n-gram와 같은 bag of word을 주로 사용하였음
- traditional method는 단어 순서나 문맥적인 정보를 무시하는 경우가 있음
- 예를 들어, *"A sunset stroll along the South Bank affords an array of stunning vantage points."* 라는 문장에서 *"Bank"* (unigram) 를 분석할 때, 우리는 이 단어가 금융기관과 둑(the land beside a river) 중 무엇을 의미하는지 모를 수 있음 -> 다의어에 대한 해석 문제
- 게다가 uppercase letter로 쓴 *"South Bank"* (bigram) 에 대해서 London에 대한 지식이 없다면 (사우스뱅크는 런던 템즈강 남부 지역 이름) 이를 금융기관으로 잘 못 받아드리게 됨 -> Named Entity에 대한 해석 문제
- 만약에 *"stroll along the South Bank"* (5-gram) 까지 보게 된다면 우리는 쉽게 뜻을 구분할 수 있게 됨
- 이처럼 더욱 복잡한 like high order n-gram 등의 feature를 사용하면 문맥적인 정보를 잘 캐치할 수 있으나, data sparsity의 문제가 생김

## Related Work

- 전통적으로 Text classification 이라는 분야의 연구는 feature engineering, feature selection and using differenct types of machine learning algorithm, 3가지 주제가 메인을 이룸
- Feature engineering을 위해 bag-of-word를 feature로 널리 사용하였고, POS tag나 tree kernel 등의 더욱 복잡한 feature들도 많이 사용됨
- Feature selection은 noisy feature를 제거하여 성능을 높이는 것을 목적으로 함 stop word 제거 방법이 이에 속함
- Machine learning algorithm은 logistic regression, naive bayes, svm 등이 있음
- 하지만 이 모든 방법들은 data sparsity problem이 있음
- 딥러닝 기반의 representation learning이 이런 sparsity problem은 해결하고자 함
- 이런 neural representation을 word embedding이라고 함

## Model

- input은 sequence of word인 document D, output은 classification 결과

![arch](https://user-images.githubusercontent.com/15166794/39083820-2051fce8-45a6-11e8-884f-04910f73788b.png)

![eq](https://user-images.githubusercontent.com/15166794/39083821-208023f2-45a6-11e8-8fb4-53ab6f1b8d45.png)

![dis](https://user-images.githubusercontent.com/15166794/39083822-20c8505a-45a6-11e8-8d14-aa94e606dac9.png)

### Word Representation Learning

- 하나의 단어는 x_i = [cl(w_i); e(w_i); cr(w_i)]로 구성되는데 각각 왼쪽 context vector, word embedding, 오른쪽 context vector이고 이를 모두 concatenate한 것임
- context vector는 위의 네트워크 그림을 참고하면 되는데 좌우의 context를 rnn 느낌으로 전파시킴
- 윈도우 범위에 해당하는 context만 사용하는 cnn과 대조적이고 더욱 명확한 단어의 의미를 반영시킬 수 있을 것이라고 봄
- 우리는 convolutional layer를 직접 사용하지는 않았지만 cnn의 역할을 할 수 있는 recurrent structure를 사용함
- 사실상 convolution도 아니고 recurrent도 아닌 두 개를 혼합한 새로운 architecture임
- 이렇게 구한 x vector는 word representation이라고 볼 수 있음

### Text Representation Learning

- 그리고 이에 대해서 한 번 더 weight를 곱하고 nonlinear function을 거쳐서 y(2) vector를 얻어냄
- y(2)에 대해서 element-wise maxpooling을 하여 차원을 유지하는 최종 y(3) vector를 구함
- 이렇게 구한 y(3)은 마치 text representation이라고 볼 수 있음
- 이렇게 하면 다양한 길이를 가지던 text들이 fixed length vector로 변함
- average pooling은 별로임. max pooling을 해야 document(text)에 숨어있는 매우 중요한 잠재적인 의미 요소(factor)를 찾아내려고 함. average를 하면 feature들이 뭉뜨그려지는 느낌
- y(3)에다가 FC를 붙이고 softmax를 취해서 최종 classification을 함

## Experiments

- 20Newsgroups, Fudan set, ACL Anthology Network, Stanford Sentiment Treebank(SST), 총 4가지 데이터 셋으로 실험을 진행함
- 비교 모델은 bag-of-word 기반의 traditional method와 neural network 기반의 모델들
- 결과는 전반적으로 traditional method보다 neural network 기반 모델들이 성능이 좋음
- 또한 convolution 기반의 neural network가 SST 데이터 셋에서 더 좋은 성능을 보이는데, 이를 통해 convolution을 하는 것이 구분력이 큰 (discriminative) feature를 뽑아내는데 더 효과적이라는 것을 알 수 있음
- cnn에는 max pooling 과정이 있기 때문에 여기서 중요한 feature들이 선별된다고 생각함
- 그리고 recursive 기반 모델들을 시간복잡도가 O(n^2)으로 우리의 모델 O(n)보다 큼. 실제로 recursive 모델은 3~5시간 걸리고 우리꺼는 몇 분 밖에 안걸림

### Contextual Information

- CNN은 문맥적인 정보를 얻기 위해서 window 기반의 convolution을 하고 우리가 제안한 RCNN의 경우는 recurrent structure를 사용함
- 때문에 cnn은 window size에 영향을 많이 받고 성능도 우리가 제안한 RCNN이 더 좋았음

### Learned Keywords

- 우리의 RCNN과 RNTN 모델을 비교
- RCNN의 경우 maxpooling에서 선택되는 단어를 추출한 것임 -> 상당히 좋은 표현 방법인듯함
- tri-gram으로 text의 keywords를 뽑은 결과가 table 3과 같음
- RNTN의 경우 전형적인 구(phrases)의 형태를 보이는데 비해 RCNN의 경우 중요한 키워드 하나와 그 양쪽의 단어를 보여줌
- 이를 통해 우리는 positive로 판별하는데 worth, sweetest, wonderful 등의 단어가 중요한 역할을 한다는 것을 알 수 있고 negative을 판별하는데 awfully, bad, boring이 중요한 역할을 한다는 것을 알 수 있음

## Conclusion

- 우리는 text classification을 위한 recurrent convolutional neural network를 제안함
- 우리의 모델은 recurrent structure를 통해서 contextual information을 잡아내고 maxpooling을 이용해 representation of text를 잡아내었음
- CNN이나 RecursiveNN보다 좋은 성능을 보임
