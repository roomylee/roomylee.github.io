---
title: "Character-Aware Neural Language Models (AAAI 2016)"
layout: post
date: 2018-03-29
headerImage: false
tag:
- character-level
- language-model
category: blog
author: roomylee
---

- Paper Link: <https://arxiv.org/abs/1508.06615>
- Author
  - Yoon Kim (Harvard University)
  - Yachine Jernite (Harvard University)
  - Divid Sontag (New York University)
  - Alexander M. Rush (New York University)
- Published at
  - AAAI 2016

---

## Abstract

- character-level input 만을 이용하는 neural language model을 제안함
- 예측(output)은 word-level에서 함
- character에 대해서 CNN, highway network를 사용하였고 이들의 output을 LSTM 기반 RNN language model (RNN-LM)에 넣어서 최종적으로 word-level 예측을 하는 것임
- 당연히 state-of-the-art보다 상당히 좋음
- 특히 morphology가 풍부한 언어들에서 더욱 좋은 결과를 보임

## Introduction

- Language modeling은 인공지능, NLP 분야에서 가장 기초적인 task 중 하나임
- 여기서 language model이란, sequence of word에 대한 확률 추정 모델이라고 볼 수 있음. 어떤 sequence가 주어졌을 때, 그 뒤에 올 단어 추정하는 모델이 이에 해당됨
- 옛날에는 n-gram 모델을 주로 사용했는데, data sparsity의 문제로 상당히 성능이 떨어졌음
- Neural language model (NLM)은 word embeddings의 방법으로 n-gram의 data sparsity 문제를 해결함
- Mikolov가 발표한 NLM 모델이 count-based n-gram language model을 능가하긴했지만, 이 역시 subword information (e.g. morphemes)를 담을 수 없었음
- 예를 들면, 사전 지식(priori) 없이 eventful, eventfully, uneventful, uneventfully가 관계가 깊다는 것을 embedding할 수 없다는 것임
- (학습 데이터에 대해) 희귀한 단어들일수록 embedding이 잘못될 가능성이 크고 이 때문에 성능의 저하가 생김
- 특히 형태소(어근)이 풍부한 언어이거나 SNS에서처럼 단어를 dynamic하게 사용하게 되면 이런 문제가 더 심각함
- 우리는 character-level CNN과 RNN-LM을 통해 subword information 활용한 language model을 제안하는 바임
- 이전 연구와 달리 morphological tagging (like POS)이나 전처리 단계가 필요없고, word embedding도 input으로 사용하지 않음
- 때문에 훨씬 적은 parameters를 사용하고, 그럼에도 불구하고 현재 state-of-the-art 보다 동등하거나 더 좋은 성능을 보임

## Model

![model](https://user-images.githubusercontent.com/15166794/37864158-2e1ecc06-2fae-11e8-8230-cf83cce548a7.png)

- 위 그림을 통해서 모델 전체 흐름을 볼 수 있는데, 현재 모델이 단어 *absurdity- 를 입력으로 받고 이전 history(as represented by the hidden state)를 조합해서 다음에 올 단어인 *is- 를 예측하는 상황임
- 첫 layer는 lookup table of character embeddings (of dimension four)을 이용해서 단어를 벡터화함
- embedding vector에 대해서 convolution을 진행함
- 그림에서 파란색은 width=2 필터 3개, width=3 필터 4개, width=4 필터 5개를 사용하였음. (왜 개수를 다르게 사용했는지는 잘 모르겠음)
- fixed-dimensional representation of the word를 얻기 위해서 convolution 결과 (feature maps)에 max-over-time pooling (=max-pooling)을 진행함
- 즉, 여기까지 한 결과를 representation of the word라고 볼 수 있는 거 같음
- 이렇게 얻은 representation of the word (vector)를 highway network에 넣음. highway network이 무엇인지는 아직 잘 모르겠음
- highway network의 output은 최종 모델인 multi-layer LSTM에 들어가고 output에 softmax를 취해 다음에 올 단어인 *is- 를 예측하는 것임. 이때는 word embedding lookup table을 사용함

### Highway Network

![eq](https://user-images.githubusercontent.com/15166794/38014035-2c1e1fb4-32a2-11e8-8120-3f300f305816.png)

- 첫 번째 식이 highway network의 모든 연산 수식임
- 의미로 보자면 input 원본이랑 fully connected layer 한 번 거친 거 중에서 더 괜찮은 걸 학습하겠다는 것이고 이 둘의 반영 비율은 t와 (1-t)로 나타남
- 반영 비율 t 역시 fully connected layer를 거쳐서 계산하게 되고 이는 두 번째 식과 같이 표현할 수 있음
- t는 비율이기 때문에 0~1의 값을 얻어야 하고 따라서 sigmoid 함수로 처리함
- fully connected의 가중치 W 벡터들(WT, WH)은 모두 square matrix임. 즉 당연한 거지만 in/out의 차원은 그대로 유지시킨다는 거임

## Experimental Setup

- Perplexity(PPL)라는 평가지표를 사용함
- PPL은 loss function으로 NLL(Negative Log Likelihood)를 사용하는데 이를 sequence 길이 T로 나누고 exp를 취한 값임

## Discussion

### Learned Word Representation

![table](https://user-images.githubusercontent.com/15166794/38018490-fc207f42-32af-11e8-86cb-089808a89e1a.png)

- 모델의 여러 층에 대한 Word Representation을 비교 및 분석하는 내용임. Table 6을 참고
- Table 6에서 before highway를 보면 *you* 의 유사단어로 *your*, *young*, *four*, *youth* 처럼 edit distance가 가까운 애들이 나옴
- 이는 표면적인 형태(비슷한 글자)를 학습했다고 볼 수 있음
- after highway를 보면 *you* 의 유사단어로 철자 상으로 거리가 있는 *we* 가 뽑힘. *while* 의 유사단어로 *though* 가 뽑히는 것도 역시 마찬가지임
- 따라서 highway layer가 이런 철자만으로는 알 수 없는 semantic feature를 학습했다고 볼 수 있음
- 다만 *his* 의 유사단어로 *hhs* 가 꼽히는 것과 같이 실수도 존재하는 것이 학습 데이터의 크기가 작긴해도 이 approach의 한계점으로 볼 수 있음
- 중요한 것은 Out-of-Vocabulary(OOV)에서 신기한 결과가 나온다는 것임
- before/after highway의 경우, *computer-aided* 나 *misinformed* 와 같은 단어의 유사단어로 품사(POS)가 같은 단어를 찾아냄
- *looooook* 의 유사단어로 *look* 을 찾는 것으로 보아, 잘못된 단어나 신조어 같이 noisy domain에 대한 text normalization으로 응용해볼 수도 있을 것 같음

### Highway Layers

![table78](https://user-images.githubusercontent.com/15166794/38020507-60f1302e-32b5-11e8-8da4-42102789e009.png)

- Table 7은 highway layer의 개수를 변화시켰을 때의 성능을 나타내는 표임. 수치가 낮을수록 좋은 거임
- One MLP Layer는 highway를 안하고 그냥 MLP를 했을 때를 의미하는 듯
- 증명할 수는 없지만(anecdotally) 실험하면서 아래와 같은 느낌을 받음
  - 하나에서 두개의 layer를 갖는 것이 중요하고, 이 이상 층을 쌓아도 그닥 성능향상은 없음. 물론 데이터 셋 크기에 따라 달라지기는 할 듯
  - max-pooling 이전에 convolutional layer는 쌓아도 별로 도움은 안됨
  - word embedding을 쓴 모델에 대해서 highway layer가 성능 향상을 가져오지 않음

### Effect of Corpus/Vocab Sizes

- Table 8은 word-level model에서 character-level model로 바꿨을 때, PPL이 감소하는 (성능이 증가하는) 비율을 corpus 크기와 vocabulary 크기에 따라 나타낸 것임. T는 코퍼스크기, |V|는 vocabulary 크기임
- vocabulary 크기를 변화시키기 위해서, most frequent k word를 사용하고 나머지는 다 <unk> 토큰으로 치환해버림. 즉 사전의 크기가 k가 되는 거임
- T=1m, |V|=100k가 비어있는 이유는 코퍼스가 작아서 100k보다 단어 개수가 적어서 그런 거임
- 표를 보면 vocabulary 크기가 크고 코퍼스 크기가 작을수록 character-level model이 더욱 성능 향상을 가져옴
- 의미를 생각해보면, 코퍼스 크기가 작을수록 word representation을 학습하기 충분치 않아서 word-level model의 성능이 안나올테고, 그에 비해 character-level model은 글자 자체를 보기에 학습 데이터가 적어도 성능 하락이 크지 않은 것 같음.
- 또한 vocabulary가 클수록 다양한 단어의 품사에서 오는 형태(ed, ing, ly)를 볼 수 있어서 character-level은 좋을 것 같고, word-level도 물론 좋긴 하겠지만 선택지가 많아지는 효과도 생겨서 분류 성능이 떨어지는 결과를 낳은 게 아닌가 싶음
- 여하튼, 모든 케이스에 대해서 character-level model이 더 좋음 (표를 보면 다 +00% 이므로)

## Conclusion

- 우리는 character-level input에 대한 neural language model을 제안함
- parameter가 적음에도 불구하고 이전 모델에 준하거나 더 좋은 성능을 보임
- CharCNN과 highway layer를 이용해서 word2vec의 한계점을 극복하는 word embedding을 할 수 있다는 가능성을 보여줌
- 모든 NLP task는 input에 대한 sequential processing of word를 해야함.
- 따라서, Encoder-Decoder model 같은 다른 model의 input에 대해 우리의 architecture를 사용하면 흥미로운 결과를 기대해볼 수 있을 거임
