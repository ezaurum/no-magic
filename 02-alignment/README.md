# 정렬과 학습 기법

사전학습 이후 모델을 제어, 파인튜닝, 정렬하는 방법. 이게 기본 모델을 뭔가 쓸모있는 걸로 바꾸는 기술임.

## 스크립트

| 스크립트        | 알고리즘                                                 | 상태     |
| -------------- | -------------------------------------------------------- | -------- |
| `microlora.py` | Low-Rank Adaptation (LoRA) 파인튜닝                       | 완료     |
| `microdpo.py`  | Direct Preference Optimization                           | 완료     |
| `microppo.py`  | RLHF를 위한 Proximal Policy Optimization (하이브리드 autograd) | 완료     |
| `micromoe.py`  | 희소 라우팅 기반 Mixture of Experts (하이브리드 autograd)   | 완료     |

### 하이브리드 Autograd 스크립트

`microppo.py`와 `micromoe.py`는 런타임 제약을 맞추기 위해 **하이브리드 autograd 접근 방식**을 사용함:

- **microppo:** 정책 모델은 스칼라 autograd (`Value` 클래스) 사용. 보상 모델과 가치 함수는 수동 그래디언트가 적용된 일반 float 배열 사용 — PPO 루프 이전에 별도로 학습됨.
- **micromoe:** 라우터는 스칼라 autograd 사용. Expert MLP는 일반 float 배열 사용 — 핵심은 라우팅 결정 메커니즘이지, expert의 순전파가 아님.

자세한 내용은 `docs/autograd-interface.md`의 표준 인터페이스와 `docs/implementation.md`의 스크립트별 상세 설명 참고.

## 추가 후보

| 알고리즘                      | 배울 수 있는 것                                          | 비고                                                              |
| ---------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| **REINFORCE**                | 베이스라인이 포함된 기본 정책 경사법                       | 더 단순한 RL 대안, PPO 이해를 위한 기초                             |
| **Dropout / Regularization** | 왜 랜덤 뉴런 비활성화가 과적합을 방지하나                  | 한 파일에 dropout, weight decay, 조기 종료를 다룰 수 있음           |
| **Batch Normalization**      | 내부 공변량 이동, 실행 통계                               | 딥 네트워크를 학습 가능하게 만든 기법                                |
| **Learning Rate Scheduling** | Warmup, cosine decay, step decay                        | 스케줄 선택이 수렴에 어떤 영향을 미치나                              |
| **Knowledge Distillation**   | 큰 모델을 모방하는 작은 모델 학습                         | 소프트 타겟을 통한 압축                                             |

## 학습 경로

이 스크립트들은 기초 단계 위에 구축됨. 권장 순서:

```
microlora.py   → 어떻게 파인튜닝이 효율적으로 작동하나 (파라미터의 1%)
microdpo.py    → 어떻게 선호도 정렬이 작동하나 (보상 모델 없이)
microppo.py    → 어떻게 RLHF가 작동하나 (보상 → 정책 전체 루프)
micromoe.py    → 어떻게 희소 라우팅이 모델 용량을 확장하나
```
