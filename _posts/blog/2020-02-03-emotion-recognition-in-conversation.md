---
title: "Emotion Recognition in Conversation: Research Challenges, Datasets, and Recent Advances"
layout: post
date: 2020-02-03
headerImage: false
tag:
- emotion-recognition-in-coversation
- emotion
category: blog
author: roomylee
---

- Paper Link: <https://arxiv.org/abs/1905.02947>
- Author
  - Soujanya Poria, Navonil Majumder, Rada Mihalcea, Eduard Hovy
- Published at
  - arXiv 2019

- Joohong's Review
  - 리뷰

---

## Abstract

- Human-like AI에 있어서 emotion을 이해하는 것은 굉장히 중요함
- 대화에서 감정을 인식하는 태스크인 Emotion Recognition in Conversation(ERC)가 최근에 점점 핫해지고 있음. 특히 사람들의 의견을 모을 수 있는 Facebook, Youtube, Reddit, Twitter 등의 public한 대화를 얻을 수 있는 플랫폼이 많아져서 더욱 힘을 받음
- 헬스케어나 교육 등의 어플리케이션 분야에도 많이 활용될 뿐 아니라, emotion-aware generation에 있어서도 굉장히 중요함
- 하지만 ERC 문제 자체가 상당히 어렵고, 연구적 챌린지가 좀 있음
- 이 논문에서는 이런 챌린지와 최근 동향에 대해서 알아보고, 각 approach들이 어떤 이유에서 성공적인 결과를 얻었을지 디스커션해봄

## 1. Introduction

- ERC는 일반적인 vanilla emotion recognition과 달리 대화의 context를 고려해야 함
- 아래의 Figure 5처럼 동일한 발화임에도 앞에 어떤 말(context)이 오느냐에 따라서 발화에 나타나는 감정이 달라지게 됨
- 따라서 presence of contextual cues, temporality in speakers' turn, speaker-specific information 등의 대화에서의 specific factor를 잘 모델링하는 것이 중요함

![figure5](/assets/images/blog/2020-02-03-emotion-recognition-in-conversation/figure5.png)

#### Task Definition

![figure2](/assets/images/blog/2020-02-03-emotion-recognition-in-conversation/figure2.png)

## 2. Research Challenges

#### a) Categorization of emotions

- Emotion은 보통 categorical 또는 dimensional, 두가지 방법 중 하나로 정의함
  - Categorical은 모델이 특정한 수의 discrete한 감정 중 하나로 분류하는 것
  - Dimensional은 continuous multi-dimensional space에 감정을 point로 표현하는 것

![figure4](/assets/images/blog/2020-02-03-emotion-recognition-in-conversation/figure4.png){: .small-image}

1. Categorical
   - Categorical 방법에서는 Plutchik의 wheel of emotion이 가장 널리 쓰이며 8가지의 감정으로 정의함
   - Ekman은 이를 anger, disgust, fear, happiness, sadness, surprise, 총 6가지 감정으로 정의함
   - 특정한 개수의 감정을 정의하는데, Ekman처럼 개수가 작으면 복잡한 감정을 표현하는 게 불가능하고, Plutchik처럼 많아지면 annotation이 어려워지게 됨
2. Dimensional
   - 보통 Valence와 Arousal이라는 2개의 축을 가진 공강에 표현함
   - Valence는 감정의 positivity, Arousal은 감정의 intensity를 표현함
   - 딱딱한 category가 아닌 연속적인 표현이 가능하고, 두 축에 대한 정보가 감정적 이해를 도울 수 있으며, 벡터에 대한 수학적 연산이 가능함
   - 반면, categorical model에 비해, 비교가 어렵다는 단점이 있음

- Dataset
  - IEMOCAP(Busso et al., 2007): categorical + dimensional
  - DailyDialogue(Li et al., 2017): categorical
  - EmoContext(Chatterjee et al., 2019): categorical (happiness, sadness, anger, others)

#### b) Basis of emotion annotation

- Emotion annotation은 annotator의 주관적 판단에 의존적임
- 이런 점을 보호하기 위해 다수의 사람이 기존 annotation을 prior로 삼아 accurate하는 작업을 하기도 함
- 실시간으로 상대방의 발화를 보고 labeling하는 게 상황에 따라 미묘하게 변하는 감정을 잡아내기는 좋지만, 현실적으로 불가능함
- EmotionLines(Chen et al., 2018)는 이미 존재하는 transcript(드라마 대본)을 이용하여 이런 문제를 완화시키기도 함
- 또한 annotator는 화자의 상황을 고려해야 함
  - 예를 들어, "Lehman Brothers stock is plummeting!!" 라는 발화에 대한 감정은 화자가 (주식을 통해) 이득을 봤는지에 따라 아주 달라질 수 있다는 것

#### c) Conversational context modeling

- 최근 contextual sentence/word embedding에 대한 state-of-the-art 연구가 많이 진행되고 있음
- 하지만 대화에서는 emotion dynamics로 인해 context representation을 만들기 어려움
- 대화에서의 Emotion dynamics는 2가지 측면이 있음
  - self dependency: 흔히 감정적 관성(emotional inertia)라고 하는 나 자신의 감정선이 대화에서 유지되는 것
  - inter-personal dependency: 대화의 상대방에 의해 감정적 영향을 받는 것
- Context modeling은 짧은 발화를 분류하는데 상당히 도움이 됨
  - "yeah", "okay" 등은 문맥을 보지 않으면 어떤 감정인지 알 수가 없음

#### d) Speaker specific modeling

-
