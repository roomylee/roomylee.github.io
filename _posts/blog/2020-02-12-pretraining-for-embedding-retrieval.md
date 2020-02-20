---
title: "[WIP] Pre-training Tasks for Embedding-based Large-scale Retrieval (ICLR 2020)"
layout: post
date: 2020-02-12
headerImage: false
tag:
- pre-training
- embedding-retrieval
- retrieval
category: blog
author: roomylee
---

- Paper Link: <https://arxiv.org/abs/2002.03932>
- Author
  - Wei-Cheng Chang, Felix X. Yu, Yin-Wen Chang, Yiming Yang, Sanjiv Kumar
  - CMU and Google
- Published at
  - ICLR 2020

---

## Abstract

- Large-scale query-document retrieval: 주어진 query(e.g., question)에 대한 set of relevant document(e.g., paragraphs containing the answer)를 large document corpus에서 찾는 문제
- 이 문제를 보통 아래의 두 단계로 품
  1. Retrieval phase: solution space를 축소해서 candidate documents의 subset을 반환함
  2. Scoring phase: Retrieval을 통해 찾은 documents를 reranking함
- Retrieval에서 핵심은 high recall과 high efficiency임
  - high efficiency: 문서 수에 대한 sublinear time 이내로 candidates 검색
- Scoring 쪽은 BERT를 통해 많은 발전을 이뤘지만, 그에 비해 retrieval은 덜 연구됨
- Retrieval 쪽 아직도 대부분의 work들이 BM25 같은 classic IR method에 의존적임
- 이런 모델들은 sparse handcrafted feature에 의존적이고 다양한 downstream task에 대해 활용하기 어려움
- 이 논문에서 우리는 보다 나은 retrieval을 위해 embedding-based retrieval models에 대한 연구를 소개할 것임
- 강력한 embedding-based model(Transformer)를 학습시키기 위해 paragraph-level pre-training tasks를 objective로 삼았음
  - Paragraph-level pre-training tasks: Inverse Cloze Task(ICT), Body First Selection(BFS), Wiki Link Prediction(WLP)
- 이를 통해 BM25를 비롯한 embedding models without Transformer보다 상당히 좋은 성능을 보였음

## 1. Introduction

## 2. The Two-Tower Retrieval Model

![figure1](/assets/images/blog/2020-02-12-pretraining-for-embedding-retrieval/figure1.png)

- 모델 구조는 심플함. 왼쪽의 Two-tower model이 제안하는 모델 구조이고, 오른쪽은 기존의 BERT와 같은 Cross-attention model임
- **Query-tower(query encoder)와 Doc-tower(document encoder)가 독립적으로 input query and candidate document를 encoding하고 두 embedding vector에 대한 inner product(cosine similarity)로 relevance를 계산함. Aggregation은 average pooling을 사용.**

#### Inference

- Cross-attention model과 비교하여 Two-tower model의 장점은 inference efficiency에 있음
- 당연한 것이지만, 모든 document embedding은 미리 계산할 수 있고(pre-computed), 주어진 unseen query $q$ 에대해서만 encoding을 하고 각 document embedding에 대한 inner product 연산만 하면 ranking을 매길 수 있음
- 반면 BERT-style의 Cross-attention models은 하나의 query가 들어올 때마다 모든 document에 대해서 inference를 해야 하기 때문에 비용이 매우 큼
- infernece time에 대한 예시
  - 128-dim embedding space에서 1000개의 query embedding과 1M document embedding 대한 inner product는 CPU 기준 수백 milliseconds 정도임
  - 하지만 cross-attention model은 GPU로 해도 수시간이 걸릴 수도 있음
- **게다가 embedding-based retrieval도 maximum inner product(MIPS) algorithm을 통해 거의 loss 없이 sublinear time에 작동할 수 있음** (Shrivastava and Li, 2014; Guo et al., 2016)

#### Learning

- 이 논문에서 training data는 "positive" query-document pairs $\tau = {(q_i, d_i)}^{|\tau|}_{i=1}$ 와 같은 형태라고 가정함
- 기본적으로 log likelihood $\max_{\theta}{\sum_{(q, d)\in\tau}{\log{p_\theta(d|q)}}}$ 를 maximizing함
- Conditional probability는 아래와 같은 Softmax로 정의됨:
  $$
  p_{\theta}{(d|q)}=\frac{\exp{(f_{\theta}(q, d))}}{\sum_{d'\in\mathcal{D}}{\exp{(f_{\theta}(q, d'))}}}
  $$
- **(각 샘플(q, d) 별로 유사도에 대한 (binary) CE가 아니고, 하나의 q에 대한 여러 d에 대해서 유사도를 구하고 이에 대한 CE라는 게 특이**)
- $\mathcal{D}$ 는 set of all possible document인데, 위 식에 따르면 Softmax 계산 비용이 상당히 큼
- 이를 완화하기위해, Sampled Softmax라는 full-Softmax의 approximation을 사용함
  - **Sampled Softmax: $\mathcal{D}$ 를 전체 document가 아니라 batch 내의 document로 구성된 작은 subset으로 치환하여 approaximation을 하는 것**
- Downstream task에 대한 $\tau$ 는 개수가 적기 때문에, 우선 set of pre-training tasks로부터 얻은 $\tau$ 에 대해 학습(pre-training)을 하고 각 downstream task에 대한 $\tau$로 fine-tuning을 함
  - pre-training tasks는 다음 3장에서 소개

## 3. Pre-training Tasks of Different Semantic Granularities

- 여기 나오는 모든 **Pre-training data는 positive query-document $(q, d)$ 쌍으로 정의된다고 가정함**
- 좋은 pre-training task의 요소
  1. Downstream task와 관련이 있어야 함
     - QA retrieval 문제를 풀려면, pre-training에서 query와 document 간의 미묘한 의미적 차이를 모델이 잡아낼 수 있게 해줘야 함
  2. 수집하는 비용이 적어야 함
     - 사람이 직접 supervision할 수 없음
- 이런 점들을 고려해서 우리는 query와 document 간의 특징적 차이를 잘 학습시키기 위한 다음의 3가지 pre-training task를 소개함
  - Inverse Cloze Task (Lee et al., 2019)
  - Body First Selection (newly proposed)
  - Wiki Link Prediction (newly proposed)
- 위의 세가지 task는 모두 manual labeling 작업 없이 Wikipedia로부터 자동으로 만듬

![figure2](/assets/images/blog/2020-02-12-pretraining-for-embedding-retrieval/figure2.png)

> $q_1$: 파랑 실선 박스,
> $q_2$: 초록 실선 박스,
> $q_3$: 빨강 실선 박스,
> $d$: 보라 점선 박스

#### Inverse Cloze Task (ICT)

- Local context within a paragraph
- 주어진 $n$ 개의 문장으로 구성된 passage $p=\{s_1, s_2, ..., s_n\}$ 에 대해서
  - 임의의 한 문장 $q=s_i, i\sim[1,n]$ 를 뽑고,
  - 나머지 문장들의 집합을 document $d=\{s_1, ..., s_{i-1}, s_{i+1}, ..., s_n\}$ 로 만들어서,
  - 최종 (positive) sample $(q, d)$ 를 만듬
- Figure 2에서 $(q_1, d)$ 에 해당됨
- Lee et al. (2019)가 제안함

#### Body First Selection (BFS)

- Global consistency within a document
- ICT와 같이 local paragraph가 아닌 외부와의 관계를 학습
- Wikipedia 특정 페이지의 첫 section에서 하나의 $q_2$ 를 뽑고, 해당 페이지의 다른 passage 중 임의로 $d$ 를 뽑음
- 첫 section에서 $q$ 를 뽑은 이유는 주로 해당 페이지의 전반적인 description이나 summary가 적혀있어서 해당 페이지의 내용을 대부분 커버할 수 있기 때문
- Figure 2에서 $(q_2, d)$ 에 해당됨
- 이 논문에서 새롭게 제안하는 task임

#### Wiki Link Prediction (WLP)

- Semantic relation between two documents
- 다른 문서 간의 semantic relation을 학습하기 위함
- 마찬가지로 Wikipedia 특정 페이지의 첫 section에서 하나의 $q_3$ 를 뽑고, 해당 페이지로 hyperlink가 걸려있는 문서 중 하나에서 document $d$ 를 뽑음
- Figure 2에서 $(q_3, d)$ 에 해당됨
- 이 역시 논문에서 새롭게 제안하는 task임

#### Masked LM (MLM)

- 기존 BERT(Devlin et al., 2019)에서의 Masked Language Modeling과 동일

## 4. Experiments

### 4.1. Experimental Setting

#### The two-tower retrieval model

- 12 layers BERT-base model
- Final embedding(512-dim) = "\[CLS\]" output에 linear 태움
- Query max length = 64
- Document max length = 288
- Pre-training
  - batch size = 8192
  - 32 TPU v3 for 100K steps
  - 2.5일동안 학습
  - Adam optimizer
    - init lr = 1e-4
    - warm-up ratio = 0.1
    - linear lr decay
- Fine-tuning
  - batch size = 512
  - 2000 training steps
  - Adam optimizer
    - lr = 5e-5

#### Pre-training tasks

![table1](/assets/images/blog/2020-02-12-pretraining-for-embedding-retrieval/table1.png){: .small-image}

- ICT, BFS, WLP 데이터는 Wikipedia corpus로부터 생성함
- 데이터 stat은 Table 1과 같고, #tokens는 WordPiece(Wu et al., 2016)로 tokenizing된 sub-words의 수를 의미함
- ICT의 경우 $d$ 를 doc-tower로 encoding할 때, article의 title과 passage를 "\[SEP\]"로 분리하여 input을 구성함

- **Two-tower Transformer를 위의 세가지 paragraph-level pro-training task에 대해서 jointly 학습을 함**
- **모두 $(q, d)$ 쌍으로 샘플이 존재하며, 각 샘플은 uniformly sampling 됨**

#### Downstream tasks

#### Evaluation

### 4.2. Main Results

### 4.3 Ablastion Study

### 4.4 Evaluation of Open-domain Retrieval

## 5. Conclusion
