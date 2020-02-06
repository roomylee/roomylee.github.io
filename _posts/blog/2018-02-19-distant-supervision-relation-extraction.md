---
title: "Distant supervision for relation extraction without labeled data"
layout: post
date: 2018-03-29
headerImage: false
tag:
- distant-supervision
- relation-extraction
category: blog
author: roomylee
---

- Paper Link: <https://web.stanford.edu/~jurafsky/mintz.pdf>

- Author
  - Mike Mintz (Stanford University)
  - Steven Bills (Stanford University)
  - Rion Snow (Stanford University)
  - Dan Jurafsky (Stanford University)
- Published at
  - ACL 2009

---

## Abstract

- Relation extraction 분야에서 ACE dataset 기반의 supervised learning은 작은 hand-labeled 코퍼스를 사용함
- 우리는 labeled data가 필요없는 새로운 방법을 제안함
- 우리는 Freebase(공개된 relation을 포함하는 semantic database임)를 기반으로 하여 distant supervision이라는 방식을 제안함
- 우리의 distant supervision 알고리즘은 supervised IE(combining 400K noisy pattern feature in a probabilistic classifier)와 unsupervised IE(extracting large numbers of relations from large corpora of any domain)의 장점을 조합한 형태임

## 1. Introduction

- 3가지 learning paradigm이 있음

1. Supervised approach
   - supervised approach는 entity와 relation에 대한 hand-labeled corpus가 존재해야 함
   - 하지만 labeled training data는 만들기가 매우 어렵고 특정 도메인 코퍼스로 학습하기에 classifier가 편향(biased)될 수 있음
2. Unsupervised approach
   - 두 번째, unsupervised approach는 entity 사이의 관계 string of word를 추출한 뒤, 그 string of word에 대해 클러스터링 및 simplification(추상화?)의 과정을 거쳐 relation class를 정의하고 이를 기반으로 dataset instance를 생성하는 방식임
   - 매우 큰 dataset을 만들 수 있지만, 위 과정에서 정의된 relation을 특정 knowledge base에서 요구하는 relation class에 mapping시키는 것이 쉽지 않다는 단점이 있음
3. Bootstrap approach
   - 마지막 세 번째, bootstrap approach는 작은 수의 seed instance(or pattern)을 사용하는 방법임
   - seed로 큰 코퍼스에서 새로운 pattern을 만들고 그 pattern으로 새로운 instance를 찾고 또 그걸로 새로운 pattern을 만들고, 이를 반복하는 방식인데 precision이 낮고 semantic drift의 단점이 있음
   - 우리는 위의 3가지 방법의 장점들이 조합된 distant supervision이라는 새로운 paradigm을 제시하려고 함
   - Distant supervision은 Freebase라는 large semantic database를 사용함
   - Distant supervision의 핵심 아이디어 중 하나는 Freebase relation으로 알려진 entity pair를 포함하는 어떤 문장이 있을 때, 해당 문장에서 entity pair는 Freebase와 동일한 relation을 갖는다고 보는 거임
   - labeled text가 아닌 (knowledge) database를 기반으로 하기에 domain에 대한 overfitting 문제에 대해 보다 자유로움

## 2. Previous work

- skip...

## 3. Freebase

- skip...

## 4. Architecture

- training step
  - 모든 instance sentence에 ner 처리를 함
  - 만약 어떤 문장이 두 entity를 포함하고 두 entity가 Freebase에서 특정 relation의 관계가 있으면, 해당 문장으로부터 feature를 추출하고 relation의 feature vector에 이를 더함
  - 여러 문장으로부터의 (relation, entity1, entity2) 형태의 tuple에 대한 feature가 모두 조합되어 더욱 풍부한 feature vector를 만듬
- testing step
  - 동일하게 ner 처리를 함
  - 이번에는 모든 entity pair가 나타나는 문장을 잠재적인 relation instance로 봄
  - entity pair가 문장에 나오면 feature를 뽑아서 그 entity pair의 feature vector에 더함
  - 예를 들어 어떤 entity pair가 test dataset에 있는 10개의 문장에 등장하고 각 문장은 3개의 feature를 추출했다고 했을 때, 그 entity pair는 총 30개의 연관된 feature를 얻게 되는 것임
  - regression classifier가 (10개 중) 각 문장에 등장한 entity pair의 relation을 예측할 때, 10개 문장 전체의 feature들을 모두 사용함
- Example 1
  - Freebase에 있는 *location-contains* relation을 생각해보자
  - 또한, 이 관계를 갖는 *<Virginia, Richmond>* 와 *<France, Nantes>* pair instance를 생각해보자
  - *'Richmond, the capital of Virginia'* 혹은 *'Henry's Edict of Nantes helped the Protestants of France'* 와 같은 문장이 있을 때, 이 문장으로부터 feature를 추출해야 함
  - Richmond sentence(첫 번째 문장)처럼 매우 유용한 문장이 있는가하면 Nantes sentence(두 번째)처럼 그닥 쓸모 없는 문장도 있음
  - testing할 때, *'Vienna, the capital of Austria'* 라는 문장을 우연히 만났다면, 이 문장의 하나 혹은 그 이상의 feature는 Richmond sentence의 feature와 매칭될 것임
  - 그리고 *<Austria, Vienna>* 가 *location-contrains* relation에 속한다는 근거를 제공할 것임
- 우리의 architecture의 가장 큰 장점 중 하나는 같은 relation을 갖는 서로 다른 많은 문장으로부터 얻은 정보를 조합할 수 있다는 것임
- Example 2
  - *<Steven Spielberg, Saving Private Ryan>* 이라는 entity pair가 있다고 하자
  - 아래의 두 문장은 *film-director* relation에 대한 문장임
    - [Steven Splielberg]'s film [Saving Private Ryan] is loosely based on the brothers' story.
    - Allison co-produced the Academy Award-winning [Saving Private Ryan], directed by [Steven Spielberg] ...
  - 첫 번째 문장은 *film-director* relation에 대한 feature로 사용할 수 있지만, *film-writer* 또는 *film-producer* relation에 대한 feature로도 사용할 수 있음
  - 두 번째 문장도 *CEO* relation(consider 'Robert Mueller directed the FBI')이라고도 볼 수 있음
- 이처럼 많은 문장들의 정보를 조합하면 새로운 의미를 이끌어낼 수도 있다는 것임

## 5. Features

![figure](https://user-images.githubusercontent.com/15166794/36368184-74c54eca-1599-11e8-812b-06513589d786.png)

### 5-1. Lexical features

- lexical features는 두 entity 주변의 단어에 대한 정보를 설명함
  - 두 entity 사이의 sequence of word
  - 그 단어들의 part-of-speech
  - entity 중 누가 먼저인가(순서)
  - entity 1의 왼쪽 k개 단어와 POS
  - entity 2의 오른쪽 k개 단어와 POS
  - k는 {0, 1, 2} 중 하나
- Penn Treebank 태그 기반의 maximum entropy tagger를 이용해 pos tagging을 함
- Penn Treebank는 문장을 syntactic/semantic tree structure로 바꾸는 프로젝트이고 거기서 사용하는 pos 태그를 이용한다는 거임

### 5-2. Syntactic features

- dependency parse tree를 이용해서 두 entity 사이의 dependency path의 tag를 이용함
- 좌우 window k 단어에 대해서도 유사한 방식으로 dependency를 feature로 사용함
- 위의 Figure 1 참고

### 5-3. Named entity tag features

- 두 entity의 Named entity tag를 feature로 사용함

### 5-4. Feature conjunction

- 위의 feature들을 결합시켜서 사용함
- 위의 Table 3을 참고하면 이해하기 쉬움

## 6. Implementation

- skip...

## 7. Evaluation

- skip...

## 8. Discussion

- Distant supervision 알고리즘은 많은 수의 relation에 대해 높은 precision으로 pattern을 추출할 수 있음
- 6, 7 section의 실험과 결과에 대한 discussion은 생략
- syntactic feature가 distant supervision에서는 확실히 유용함
- Future work를 아래와 같이 제시함
  - chunk-based syntactic feature가 full parsing의 오버헤드를 줄일 수 있을 것
  - coreference resolution이 성능향상을 가져올 것
