# 기초

최신 AI 시스템을 구성하는 핵심 알고리즘. 이게 근본임. 이걸 이해했으면 나머지는 조합.

## 스크립트

| 스크립트             | 알고리즘                                                 | 상태      |
| ------------------- | -------------------------------------------------------- | --------- |
| `microgpt.py`       | 스칼라 autograd를 사용한 자기회귀 언어 모델 (GPT)          | 완료      |
| `micrornn.py`       | Vanilla RNN vs. GRU — 기울기 소실과 게이팅                | 완료      |
| `microtokenizer.py` | Byte-Pair Encoding (BPE) 토크나이제이션                   | 완료      |
| `microembedding.py` | 대조 학습 기반 임베딩 (InfoNCE)                           | 완료      |
| `microrag.py`       | Retrieval-Augmented Generation (BM25 + MLP)              | 완료      |
| `microdiffusion.py` | 2D 포인트 클라우드에 대한 디노이징 디퓨전                  | 완료      |
| `microvae.py`       | Reparameterization trick을 사용한 Variational Autoencoder | 완료      |

## 추가 후보

아래 알고리즘은 향후 추가 가능성이 높은 후보임. 각각 프로젝트 제약 조건(단일 파일, 의존성 없음, 학습과 추론 포함, CPU에서 7분 이내)을 충족해야 함.

| 알고리즘                               | 배울 수 있는 것                                          | 비고                                                                      |
| -------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| **LSTM**                               | Long Short-Term Memory 게이팅 (3개 게이트 vs. GRU의 2개) | micrornn.py를 확장하거나 독립 스크립트로 가능                                |
| **GAN**                                | 2D 데이터에 대한 생성적 적대 신경망                        | 생성자 vs. 판별자 역학, 모드 붕괴, 학습 불안정성                             |
| **Transformer Encoder (BERT 스타일)**   | 마스크 언어 모델링, 양방향 어텐션                          | microgpt (디코더 전용)와 대비됨                                             |
| **ConvNet**                            | 작은 이미지에 대한 합성곱 밑바닥 구현                      | 커널 슬라이딩, 풀링, 피처 맵 — 비전의 기본 요소                              |
| **Optimizer Comparison**               | SGD vs. Momentum vs. Adam 비교                           | 수렴 역학, 적응형 학습률                                                    |
| **Word2Vec**                           | 네거티브 샘플링을 사용한 Skip-gram                        | 고전적 임베딩 알고리즘, 대조 학습보다 단순함                                  |

## 학습 경로

기초 단계를 순서대로 학습하려면 아래 순서를 따를 것:

```plaintext
microtokenizer.py   → 어떻게 텍스트가 숫자가 되나
microembedding.py   → 어떻게 의미가 기하학이 되나
microgpt.py         → 어떻게 시퀀스가 예측이 되나
microrag.py         → 어떻게 검색이 생성을 보강하나
micrornn.py         → 어텐션 이전에 시퀀스를 어떻게 모델링했나
microdiffusion.py   → 어떻게 노이즈에서 데이터가 나오나
microvae.py         → 어떻게 압축된 생성 표현을 학습하나
```
