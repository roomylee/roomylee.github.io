---
title: "Literature mining of host–pathogen interactions: comparing feature-based supervised learning and language-based approaches (Bioinformatics 2012)"
layout: post
date: 2018-03-03
headerImage: false
tag:
- literature-mining
- host-pathogen-interaction
- feature-based-supervised-learning
- language-based-supervised-learning
category: blog
author: roomylee
---

- Paper Link: <https://www.ncbi.nlm.nih.gov/pubmed/22285561>
- Author
  - Thanh Thieu (University of Missouri)
  - Sneha Joshi (University of Missouri)
  - Samantha Warren (University of Missouri)
  - Dmitry Korkin (University of Missouri)
- Published at
  - Bioinformatics 2012

---

## Abstract

### Motivation

- 전염병(infectious disease)에서 숙주(host)와 병원균(pathogen) 간의 관계(host-pathogen interations; HPIs)가 상당히 중요하다.
- 타켓하는 질병 또는 숙주 유기체(host organism)에 대한 다양한 데이터 베이스가 존재하고 HPI는 여러 데이터 베이스에 걸쳐서 나타난다.
- Biomedical literature 로부터 자동으로 HPI를 추출하는 방법은 이런 데이터 베이스(repository)를 만드는데 매우 중요하다.

### Results

- 우리는 2가지 새로운 approach를 제안
  1. PubMed에 있는 title 또는 abstract가 HPI data를 포함하는지 여부를 찾아내는 것
  2. 유기체와 단백질 간의 상호작용 정보를 추출해내는 것

- 첫 번째 approach는 SVM을 이용한 feature-based supervised learning 방법이다.
- 각 문장으로부터 host/pathogen organism의 이름, 단백질(protein), 유전자(gene), HPI를 나타내는 키워드, protein-protein interaction (PPI) 등을 feature를 추출하여 SVM을 학습시킨다.
- 두 번째는 language-based 방법인데, link grammar parser와 training example로 부터 생성한 semantic pattern을 조합한다.
- 직접 만든 HPI data를 기반으로 train과 test를 진행하였다.
- Classification task에 대해 기존 PPI에 대한 approach보다 accuracy와 recall 면에서 좋았다.

## 1. Introduction

- HPI를 추출하기 위한 다양한 approach가 제안되었으나, 아직 biomedical literature로부터 molecular HPI data를 추출하기 위한 완전 자동 시스템은 만들어진 바가 없다.
- Biomolecular 정보에 대한 방법은 아래의 3가지로 크게 나눌 수 있다.
  1. text에서 protein or gene identification하기
  2. protein과 gene을 literature 기반 함수적으로 annotation하기
  3. biological molecule 간의 관계 정보 추출하기 (ex. protein-RNA or gene)

- 세 번째 방법으로 찾은 관계는 gene과 protein이 text에서 동시에 나타나는 경우부터 PPI 검출 및 신호전달 네트워크와 대사 경로 확인까지 이른다.
- HPI에 비해 PPI가 보다 많은 연구가 이루어져 있다.
- PPI를 찾기 위한 가장 베이직한 방법은 protein이나 gene이 같은 문장에 동시에 나타나면 PPI가 있다고 보는 것이다.
- 보다 발전된 방법은 의미적 구조를 잡아내기 위한 pattern matching 기법이다.
- 이런 패턴은 수동으로 만들거나 dynamic programming 등을 이용해서 자동으로 만들어낸다.
- 패턴 말고 또다른 방법은 바로 feature 기반의 머신러닝 기법이다.
- link grammar, context-free grammar 등 dictionary rule 기반의 feature를 사용하여 관련된 단어 pair를 찾아내는 방식이 SOTA이다.
- 최근 PPI mining을 위한 subtask는 다음과 같다.
  1. PPI 관련 문서의 classification
  2. PPI가 있는 문장을 identification
  3. interaction이 있는 protein pair를 identification

- 위 subtask에 대해 머신러닝 기반의 방법들을 시도해보고 있다.
- 우리는 PubMed publication의 title과 abstract로부터 HPI text mining을 위한 2가지 approach를 제안한다.
- 첫 번째는 SVM 기반의 supervised learning feature-based approach
- 두 번째는 link grammar를 이용한 language-based approach

## 2. Methods

- 우리는 아래의 3가지 subtask를 정의했다.

  1. Task 1: biomedical publication의 title과 abstract이 합쳐진 expanded abstract이 주어지면, *HPI-relevant* 인지 결정. *HPI-relevant* 란 host과 pathogen의 interaction과 그 이름과 같은 HPI 정보가 나타나는지를 의미한다.
  2. HPI 정보를 포함하는 expanded abstract, 즉 *HPI-relevant* 인 text에 대해서 정확히 어떤 문장에 HPI 정보가 나타나는지를 결정하는 것.
  3. interaction에 관여하는 특정 host와 pathogen pair를 *HPI-relevant* abstract에서 찾아내는 것.

### 2.1 Feature-based approach

- feature-based approach는 5가지 단계로 이루어진다.
  1. abstract에서 protein/gene과 (host) organism을 찾아서 tagging한다. (NER)
  2. abstract으로부터 feature vector를 생성해낸다.
  3. 해당 abstract이 *HPI-relevant* 한지 수동으로 labeling하고, feature vector를 input으로 해당 label을 output으로 두어서 이 abstract의 *HPI-relevant* 여부를 분류하도록 supervised learning을 한다.
  4. 만약 *HPI-relevant* 하다면, (1)abstract에서 어떤 문장이 HPI 정보를 가지고 있는지를 찾고 (2)이 정보가 얼마나 확실한지 알려주고 (3)그 문장에서 protein/gene과 (host) organism의 정보를 추출해준다.
  5. 학습을 마친 시스템에 대해서 testset으로 평가한다.

- Text preprocessing
  - 먼저 한줄씩 잘랐음
  - 'i.e.', 'e.g.', 'vs.' 같은 약어의 period(.)는 다 공백으로 바꿔준 뒤 period를 기준으로 잘랐다.
  - Entity tagging을 위해서 NLProt라는 NER tool을 사용했다. 이 tool에 대한 설명은 생략
- Support vector machines
  - 이 문제는 간단히 생각해보면 abstract을 N개의 feature vector로 변환시키고 이를 input으로 하여 HPI 정보가 있는지 여부를 binary classification (y={-1,1}) 하는 것이다.
  - 우리는 이를 위해 supervised learning model 중 하나인 SVM을 사용했다.
- Feature vectors
  - 각 abstract은 12 차원의 feature vector로 변환된다.
  - 대부분의 feature는 HPI topic과 관련된 keyword를 달려있다.
  - feature x1과 x2는 abstract에 host와 pathogen이 존재하는지를 나타낸다. 이 feature들은 NER 결과에 기반하여 결정된다.
  - feature x3과 x4는 태깅된 host와 pathogen이 나타낸 횟수를 의미한다.
  - feature x5는 binary feature이며, PPI keyword의 존재 여부를 나타낸다.
  - feature x6과 x7은 PPI(HPI아닌가..?) keyword의 개수에 대한 통계 정보를 나나내는데, 각각 전제 단어 중 interaction keyword 수에 대한 백분율, 전체 문장 수 중 interaction keyword를 포함하는 문장의 수를 나나낸다. interaction keyword는 사전으로 정의되어 있다.
  - feature x8은 typicality of each keyword를 나타내는데, typicality of keyword란 해당 keyword를 포함한 abstract의 개수를 의미한다.
  - feature x9는 abstract에서 experimental keyword의 개수를 의미한다. experimental keyword는 사전으로 정의되어 있다.
  - feature x10은 abstract에서 전체 단어 수 중 negative keyword의 개수에 대한 백분율으로 정의한다.
  - feature x11은 abstract에서 negative keyword가 HPI 정보에 쓰였는지에 대한 여부를 나타낸다. 이는 한 문장에서 interaction keyword와 negation keyword 사이의 단어 수로 정의한다.
  - feature x12는 HPI-specific keyword에 대한 것이고, 이는 전체 abstract에 나타난 단어의 개수에 대한 해당 keyword의 비율(백분율)로 정의한다.
- Supervised training and classification using SVM
  - SVM을 이용해 abstract이 *HPI-relevant* 인지 아닌지를 학습시킨다.
  - 학습된 SVM을 두 번에 걸쳐서 사용하게 된다.
  - 첫 번째는 abstract이 *HPI-relevant* 한 지 아닌지를 판별하는데 사용되고, 두 번째는 만약 relevant한 abstract에 대해서 어떤 문장이 *HPI-relevant* data를 가장 많이 포함하고 있을 지 결정하는데 사용된다.
  - 두 번째의 경우, 각 문장 별로 feature vector를 생성해내고 이를 SVM의 input으로 넣는다.
  - SVM의 accuracy는 보통 최적화할 수 있는 parameter의 수에 달려있다.
  - 우리는 두개의 parameter C와 gamma를 사용하였다.
- Handling information uncertainty
  - SVM으로 분류된 하나의 abstract에서 나온 HPI data를 갖고 있는 문장 중, 한 문장이라도 interaction keyword 앞에 uncertainty keyword가 나타나는 경우 uncertainty하다고 판별하고 결과에서 제한다.

### 2.2 Language-based approach

- 두 번째 approach는 language-based formalism에 대한 것인데 구체적으로는 link grammar 라는 것을 사용한다.
- 우리의 approach는 PPI를 정보를 추출하는 language-based system과 유사하지만 pipeline에 새로운 모듈을 추가해주어야 한다.
- Method organization
  - HPI mining pipeline은 8가지 스텝으로 구성된다.
    1. text preprocessing
    2. entity(host, pathogen) tagging
    3. grammar parsing (dependency structure)
    4. anaphora resolution (대명사 처리)
    5. syntactic extraction (복잡한 문장을 간단한 문장으로 쪼갬)
    6. role matching (semantic role에 대한 결정)
    7. interaction keyword tagging
    8. HPI information 추출
  - feature based approach와 다르게, language-based approach는 Task 2와 3 (HPI 포함 문장 찾기, 문장에서 host와 pathogen pair 및 interacting protein/gene 찾기)에서 보다 직접적으로 다룬다.
- Entity tagging
  - feature-based approach보다 세세하게 entity tagging을 진행함
  - 아래의 3단계를 거침
    1. NLProt를 이용한 pretein/gene tagging (feature-based approach에서 이용한 방법)
    2. host/pathogen organism dictionary-based matching
    3. post-precessing
  - NLProt를 적용시키고 UniProt를 이용해서 synonym에 대한 그룹핑 처리를 한다. 이게 1, 2단계
  - synonym은 NCBI Taxonom IDs를 이용해서 처리하였다.
  - post-processing 스텝에서는 mutual context 정보를 detection accuracy 향상을 위해 사용했다.
  - 우리 시스템은 (1)사전에 없는 추가적인 host/pathogen 정보를 찾고 (2)protein/gene을 올바른 organism에 다시 할당하기 위해서 link grammar가 제공하는 phrase structure를 사용했다.
  - 그리고 해당 structure에 대해서 아래와 같은 패턴을 이용했다.
    1. Organism name + protein name (e.g. 'Arabidopsis RIN4 protein')
    2. Protein name + preposition + organism name (e.g. 'RXLX of human')
  - 예를 들어, 'Arabidopsis RIN4 protein' 같은 경우, NLProt를 사용하면 RIN4를 pathogenic organism이라고 할텐데, dictionary matching을 하면 host organism이라고 할 것이고 이에 대해 post-processing을 하면 해당 구(phrase)를 pattern 1로 처리하여 host protein인 Arabidopsis를 RIN4를 포함하는 organism으로 볼 것이다.
- Link grammar parsing
  - link grammar는 dependency를 내포하는 context-free grammar이다.
  - 룰 기반으로 관련된 단어 pair들을 link 걸어준다.
  - open source인 link grammar parser를 사용했다.
  - biomedical에 costomize된 BioLG 라는 것과 English-language semantic dependency replationship extractor인 RelEx 라는 것도 추가 feature로 사용하였다.
- A three-layer entity framework (BioLG?)
  - three-layer entity framework은 entity tagging module을 위해서 만들어졌다. Fig 3 참고
  - 가장 아래층에는 UniProt나 NCBI Taxonomy ID에 의해서 정의된 real entity 집합으로 이루어져있다.
  - 중간층은 abstract에 있는 모든 문장들의 집합으로 이루어져있다.
  - 이 중간층에서 각 textual entity는 유일한 real entity에 매핑된다.
  - 가장 위층은 각 문장들로부터 선택된 가장 좋은 link grammar parse로 구성된다.
  - 하나의 문장에서 다수의 link grammar parse가 나올 수 있으며, 하나 혹은 그 이상의 link grammar node가 하나의 textual entity에 연결된다.
- Anaphora resolution (RelEx?)
  - 이 모듈에서는 대명사의 의미를 결정하는 작업을 한다.
  - interaction은 여러 문장에 걸쳐서 대명사를 사용하며 나타나기 때문에 anaphora resolution은 매우 중요하다.
  - RelEx라는 anaphora resolution module을 사용했다.
- Syntactic extraction
  - 복문 등의 복잡한 문장이 종종 있는데 우리 시스템의 syntactic extractor module이 simple sentence를 찾아준다.
  - 이 시스템은 'The Pseudomonas syringae type III effector protein avirulence protein B (AvrB) is delivered into plant cells, where it targets the Arabidopsis RIN4 protein' 를 'The Pseudomonas syringae type III effector protein avirulence protein B (AvrB) is delivered into plant cells'와 'The Pseudomonas syringae type III effector protein avirulence protein B (AvrB) targets the Arabidopsis RIN4 protein', 두 개의 심플한 문장으로 분리시켜 준다.
- Interaction keyword tagging
  - interaction keyword를 tagging하기 위해 WordNet을 이용해서 stemming을 한다.
  - 수동으로 만든 interaction keyword dictionary를 기반으로 stem dictionary를 만든다.
  - 이를 이용해서 stemming을 진행한다.
- Role type matching
  - 이 모듈에서는 subject, verb, object, modifiying phrase 등의 각 문법적 요소(syntactic component)의 역할이 결정된다.
  - 하나의 host, pathogen entity 혹은 interaction keyword 등은 elementary type으로, 두 개면 partial, 세 개면 complete으로 role의 타입을 나눴다. (왜 나눈거지...)
- Interaction extraction
  - 아래와 같은 패턴을 이용해서 host와 pathogen의 interaction을 추출해낸다.
    1. A + interaction verb + B
    2. Interaction noun + 'between' + A + 'and' + B
    3. Interaction noun + 'of' + A + 'by' + B
  - 각각은 syntactic component(subject, verb, object 등)라고 볼 수 있다.
- Uncertainty analysis
  - negation keyword('does not', 'cannot')과 uncertainty keyword('possibly', 'may') 사전을 이용하여 불확실한 정보를 걸러낸다.
- Interaction normalization
  - 여러 문장에 중복되어 나타나는 HPI를 찾아내는 것이다.
  - 같은 tuple로 나타나는 host/pathogen/protein/gene에 대해서 중복으로 처리한다.
  - 이미 Entity tagging 단계에서 protein/gene과 organism의 이름은 real entity로 normalize 되었기에, real entity 단에서 중복체크를 한다.
  - HPI가 여러 줄에 나타날 때, 그 중 하나라도 negative하면 해당 HPI를 negative로 처리하고 uncertainty도 마찬가지이다. 만약 이 두 조건에 해당하지 않을 때만 certain HPI로 본다.

### 2.3 Assessment

- 2가지 방법으로 평가했다. 하나는 서로 비교하는 방법, 그리고 하나는 PPI SOTA에 기반한 naive protocol.
- Naive protocol
  - 이 방법은 PPI를 이용해서 HPI 문장을 추출하는 방식이다. 평가하는 방식이 아닌 듯.
  - (1) PPI를 기준으로 abstract이 PPI 정보를 갖고 있는지 보고, (2) PPI 중 최소 하나의 host, 하나의 pathogen keyword를 포함하고 있으면 HPI를 포함하고 있다고 보는 것이다.
- Assessment of approaches
  - naive approach를 포함한 총 3개의 approach (위에서 소개한 두개)의 성능은 아래와 같이 평가 된다.
    - Task 1 : abstract이 HPI-relevant한 지 binary classification
      - accuracy, precision, recall, F-score, AUC(area under ROC curve)총 5가지의 지표를 사용해서 평가한다. AUC 같은 경우 feature-based method에 대해서만 한다.
      - feature-based approach의 경우 3가지 방식으로 지표를 구하게 되는데, 우선 350개의 abstract 중 75%인 262개의 abstract은 train으로 쓰고 나머지는 test로 쓴다.
      - 첫 번째 방식은 위에서 분할한 testset으로 평가를 진행하는 것이다.
      - 두 번째 방식은 trainset에 대해서 10-fold CV를 진행하는 것이다.
      - 마지막 세 번째 방식은 train과 test를 합쳐서 leave-one-out CV를 진행하는 것이다.
      - language-based와 naive approach 역시 testset에 대해서 평가를 진행하고 feature-based와 비교한다.
    - Task 2 : HPI-relevant abstract에서 어떤 문장이 HPI-relevant information을 담고 있는지
      - 두가지 종류로 sentence를 annotation 했다.
      - 첫 번째는 host와 pathogen 모두가 한 문장에 나타나면 complete, 여러 문장에 이 정보가 나뉘어져 있으면 partial이라고 annotation 한다.
      - 다음의 2가지 평가 지표를 사용한다.
        1. prediction accuracy를 보기 위한 HPI-relevant라고 추출된 문장 중 tp sentence의 비율
        2. prediction coverage를 보기 위한 positive으로 annotation된 문장 중 실제로 그렇게 예측된 비율
      - 또한 위의 두 지표를 다음 4가지 set에 대해서 평가한다.
        1. language-based model에 의해서 both organism이 추출된 abstract에 있는 complete sentence들의 집합
        2. language-based model에 의해서 both organism이 추출된 abstract에 있는 partial sentence들의 집합
        3. language-based model에 의해서 both protein/gene이 추출된 abstract에 있는 complete sentence들의 집합
        4. language-based model에 의해서 both protein/gene이 추출된 abstract에 있는 partial sentence들의 집합
      - 그래서 최종적으로 8개의 measure가 각 approach마다 나온다.
    - Task 3 : 문장에서 host-pathogen pair interaction 맞추기
      - host, pathogen pair를 찾았는지, organism을 찾았는지에 대한 precision, recall, f-score를 본다.
      - 따라서 6가지 지표가 나오고, protein/gene과 organism 중 하나 이상만 나타나는 경우에 대해 또 6가지 지표를 본다.

## 3. Results

### 3.1 Data collection

- MEDLINE/PubMed database에서 데이터를 수집했다.
- data는 29 host와 77 pathogen organism 사이에 PPI 정보를 포함한 175 positive abstract와 HPI 정보가 없는 175 negative abstract으로 구성되어 있다.
- positive set은 최소 하나의 host와 하나의 pathogen을 포함하는 abstract로 구성된다.
- 또한, 각 abstract는 host 이름, pathogen 이름, host protein/gene, pathogen protein/gene, certain/uncertain HPI에 대해서 annotation된다.
- negative set도 positive과 유사하게 만든다.

### 3.2 Evaluation of feature-based, language-based and naive approaches

- precision, recall, f-score, AUC 등을 봄
