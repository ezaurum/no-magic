# 시스템과 추론

모델을 빠르고, 작고, 배포 가능하게 만드는 엔지니어링. 연구 프로토타입을 프로덕션 시스템으로 바꾸는 최적화를 파헤침.

## 스크립트

| 스크립트             | 알고리즘                                                          | 상태     |
| ------------------- | ---------------------------------------------------------------- | -------- |
| `microattention.py` | 어텐션 변형 모음 (MHA, GQA, MQA, sliding window)                   | 완료     |
| `microkv.py`        | KV-cache 메커니즘 (있을 때 vs. 없을 때, paged attention)            | 완료     |
| `microquant.py`     | 가중치 양자화 (INT8, INT4, per-channel vs. per-tensor)             | 완료     |
| `microflash.py`     | Flash Attention 알고리즘 시뮬레이션 (타일링, online softmax)        | 완료     |
| `microbeam.py`      | 디코딩 전략 (greedy, top-k, top-p, beam, speculative)              | 완료     |

### 순전파 전용 스크립트

`microattention.py`와 `microflash.py`는 **순전파 비교** 스크립트임 — 모델 학습을 하지 않음. 이건 학습+추론 규칙의 의도적 예외임: 교육적 가치는 구현을 나란히 비교하는 데 있지, 학습을 보여주는 데 있는 게 아님.

### 알고리즘 시뮬레이션

`microflash.py`는 Flash Attention의 **알고리즘 시뮬레이션**임. 순수 Python은 표준 어텐션보다 느릴 수밖에 없음. 이 스크립트는 Flash Attention이 _무엇을_ 하는지(타일 연산, online softmax)를 보여주는 거지, _왜_ 실제로 빠른지(GPU 메모리 계층 최적화)를 보여주는 게 아님. 주석에서 이 차이를 명시함.

## 추가 후보

| 알고리즘                                | 배울 수 있는 것                                            | 비고                                                     |
| -------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **Speculative Decoding (독립 버전)**    | Draft-verify 패러다임 심층 분석                              | 현재 microbeam에 포함됨; 별도 심층 분석 가능               |
| **Model Parallelism**                  | 텐서 병렬화, 파이프라인 병렬화 개념                          | 분산 추론의 알고리즘 시뮬레이션                             |
| **Continuous Batching**                | 처리량 최적화를 위한 동적 배칭                               | vLLM 성능의 핵심 기법                                     |
| **Prefix Caching**                     | 공통 프리픽스를 가진 요청 간 KV-cache 공유                   | microkv 개념의 확장                                       |
| **Activation Checkpointing**          | 학습 중 연산과 메모리 트레이드오프                            | 그래디언트 체크포인팅 밑바닥 구현                           |
| **Mixed Precision**                    | 손실 스케일링을 포함한 FP16/BF16 학습                        | 반정밀도 학습의 작동 원리                                  |

## 학습 경로

이 스크립트들은 아무 순서로 학습해도 되지만, 아래 순서가 개념을 점진적으로 쌓아감:

```
microattention.py   → 어텐션이 실제로 어떻게 작동하나 (모든 변형)
microkv.py          → 왜 LLM 추론이 메모리 바운드인가
microflash.py       → 어텐션이 어떻게 빨라지나 (타일링 + online softmax)
microquant.py       → 모델이 어떻게 압축되나 (INT8/INT4)
microbeam.py        → 디코딩 전략이 출력 품질을 어떻게 결정하나
```
