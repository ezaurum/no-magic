# 마술아님 — 구현 계획

이 문서는 `no-magic` 저장소의 모든 스크립트를 상세히 기술함: 각 스크립트가 가르치는 것, 구현하는 것, 아키텍처 결정, 데이터셋 전략, 예상 복잡도. 컬렉션을 구축하기 위한 엔지니어링 스펙으로 사용할 것.

### Karpathy의 작업과의 관계

이 프로젝트는 Andrej Karpathy의 [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore), `microgpt.py`에서 영감을 받았음. 그의 작업을 참조하고 출처를 밝히지만 복제하지는 않음. 구체적으로: `microgpt.py`는 완전한 출처 표기와 함께 포함됨; `micrornn.py`는 makemore가 여러 노트북에 걸쳐 탐구하는 RNN → GRU 진행을 하나의 비교 파일로 압축함; autograd 엔진(micrograd)은 이미 `microgpt.py` 안에 내장되어 있음. 해당 주제에 대한 심층 학습은 Karpathy의 원본 저장소를 참조할 것.

---

## 저장소 구조

```plaintext
no-magic/
├── README.md
├── CONTRIBUTING.md
├── docs/
│   ├── implementation.md       # 이 파일 — 엔지니어링 스펙
│   └── autograd-interface.md   # 표준 Value class 인터페이스
├── 01-foundations/
│   ├── README.md               # 알고리즘 목록 + 로드맵
│   ├── microgpt.py
│   ├── micrornn.py
│   ├── microtokenizer.py
│   ├── microembedding.py
│   ├── microrag.py
│   ├── microdiffusion.py
│   └── microvae.py
├── 02-alignment/
│   ├── README.md               # 알고리즘 목록 + 로드맵
│   ├── microlora.py
│   ├── microdpo.py
│   ├── microppo.py
│   └── micromoe.py
└── 03-systems/
    ├── README.md               # 알고리즘 목록 + 로드맵
    ├── microattention.py
    ├── microkv.py
    ├── microquant.py
    ├── microflash.py
    └── microbeam.py
```

## 설계 제약 (모든 스크립트에 적용)

| 제약              | 규칙                                                                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 파일 수           | 알고리즘당 정확히 하나의 `.py` 파일                                                                                                                                                      |
| 의존성            | Python 표준 라이브러리만 (`os`, `math`, `random`, `json`, `struct`, `urllib`, `collections`, `itertools`, `functools`, `string`, `hashlib`, `time`)                                       |
| 실행              | `python script.py`를 인자 없이 실행하면 전체 학습 + 추론 루프가 돌아감                                                                                                                   |
| 런타임            | **M-series Mac에서 7분 이내** 또는 **2019년대 Intel i5에서 10분 이내**. 느린 하드웨어를 위한 여유를 두고 7분을 목표로 함.                                                                  |
| 데이터셋          | 첫 실행 시 `urllib`로 자동 다운로드, 로컬 캐시, 5MB 이하                                                                                                                                 |
| 출력              | 학습 진행 상황과 추론 결과를 stdout에 출력                                                                                                                                               |
| 시드              | 재현성을 위해 `random.seed(42)`                                                                                                                                                          |
| 주석              | **필수.** 모든 스크립트는 `CONTRIBUTING.md`의 주석 표준을 따라야 함. 충분한 주석 없이는 병합되지 않음.                                                                                     |
| Autograd          | scalar autograd를 사용하는 스크립트는 `docs/autograd-interface.md`에 정의된 표준 `Value` class 인터페이스를 구현해야 함                                                                   |
| 수치 안정성       | 모든 스크립트는 stable softmax (`exp(x - max(x))`), clipped log-probability (`max(p, 1e-10)`), Adam epsilon (`1e-8`)을 사용해야 함. 필수 패턴은 `docs/autograd-interface.md` 참조.        |

### 최소 하드웨어 요구사항

- Python 3.10+
- 8 GB RAM
- 모든 최신 CPU (2019년대 이후)

스크립트는 M-series Mac(기본)과 Intel i5(보조)에서 테스트됨. M-series에서 7분 안에 실행되면, 2019년대 Intel에서 10분 이내에 완료되어야 함.

### 주석 표준

전체 주석 표준(필수 주석 7종, 밀도 목표, 예시)은 `CONTRIBUTING.md`를 참조할 것. 이것이 주석 품질의 유일한 권위 있는 참조임.

**요약:** 파일 테시스, 섹션 헤더, "왜" 주석, 수식-코드 매핑, 직관 주석, 표지판 주석, 자명한 주석 금지. 30-40% 주석 밀도를 목표로 함. 테스트: _의욕 있는 엔지니어가 이 파일을 처음부터 끝까지 한 번에 읽고 알고리즘을 이해할 수 있는가?_

### Autograd Callout 패턴

scalar autograd `Value` class를 재구현하는 스크립트(microgpt, micrornn, microlora, microdpo, microppo, micromoe)는 Value class 정의 직후에 callout 블록을 포함해야 함:

```python
# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# This Value class follows the canonical interface (see docs/autograd-interface.md)
# with the following additions/modifications for [algorithm name]:
# - sigmoid(): Required for GRU gating (not in microgpt's base set)
# - clip(): Required for PPO ratio clipping
# See docs/autograd-interface.md for the full canonical interface.
```

이렇게 하면 독자가 autograd 섹션을 건너뛰고 스크립트별 차이를 놓치는 것을 방지함.

---

## 01 — 기초

### `microgpt.py` — 자기회귀 언어 모델

> _"GPT를 순수한, 의존성 없는 Python으로 학습하고 추론하는 가장 원자적인 방법."_

**가르치는 것:**

- Reverse-mode automatic differentiation을 통한 scalar autograd
- Token 및 positional embedding
- Causal masking을 사용한 multi-head self-attention (점진적 KV 구성 방식)
- RMSNorm, residual connection, MLP 블록
- Cross-entropy loss, bias correction이 포함된 Adam optimizer
- Temperature-scaled 자기회귀 샘플링

**아키텍처:** GPT-2 변형 — LayerNorm 대신 RMSNorm, 바이어스 없음, GELU 대신 ReLU

**데이터셋:** Karpathy의 makemore에서 가져온 `names.txt` (~32K 이름, 18KB, urllib로 자동 다운로드)

**하이퍼파라미터:** `n_embd=16, n_head=4, n_layer=1, block_size=16, lr=0.01, ~4,200 params`

**성공 기준:**

- 최종 loss: < 2.7 (문자 단위 cross-entropy, 스텝당 단일 문서 — 확률적, 평균 아님)
- 생성된 이름: 50% 이상이 발음 가능한 영어 유사 시퀀스
- 런타임: M-series Mac에서 < 7분

---

### `micrornn.py` — 순환 시퀀스 모델링

> _"어텐션이 모든 것을 정복하기 전 — 순환으로 시퀀스를 모델링한 방법, 그리고 게이팅이 왜 돌파구였는지."_

**가르치는 것:**

- Vanilla RNN: 시퀀스 히스토리의 손실 있는 압축으로서의 은닉 상태
- Backpropagation through time (BPTT): 기울기 계산을 위한 순환 펼치기
- 기울기 소실 문제: vanilla RNN이 긴 시퀀스에서 왜 실패하는지 (말로만 설명하는 게 아니라 수치적으로 시연)
- GRU 게이팅: reset gate (무엇을 잊을지)와 update gate (무엇을 유지할지)
- 게이팅이 기울기 문제를 해결하는 이유: update gate가 기울기 고속도로를 만듦
- RNN → GRU/LSTM → Transformer의 역사적 흐름, 그리고 각 전환에서 얻은 것

**알고리즘 개요:**

```
1. scalar autograd 구현 (Value class 패턴 재사용)
2. Vanilla RNN 구현:
   a. h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
   b. y_t = W_hy @ h_t + b_y
   c. 이름 생성으로 학습, loss 곡선 출력
   d. 각 타임스텝에서 기울기 norm 출력 — 지수적 감소를 보여줌
3. GRU 구현:
   a. z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})      # update gate
   b. r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1})      # reset gate
   c. h_candidate = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}))
   d. h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate
   e. 같은 데이터로 학습, loss 곡선 출력
   f. 기울기 norm 출력 — 게이팅을 통한 안정적 기울기를 보여줌
4. 비교: 최종 loss, 기울기 건강도, 생성된 이름 품질
5. 추론: 양쪽 모델에서 이름을 나란히 생성
```

**데이터셋:** `names.txt` — 아키텍처 간 직접 비교를 위해 `microgpt.py`와 같은 태스크.

**핵심 구현 사항:**

- 두 모델 모두 동일한 데이터와 동일한 하이퍼파라미터로 학습 — 유일한 변수는 아키텍처
- 기울기 norm 추적이 핵심 교육 도구: 샘플 시퀀스에 대해 각 타임스텝의 `||dL/dh_t||`를 출력하여 기울기 소실을 이론이 아닌 눈으로 확인 가능하게 함
- Sigmoid은 autograd 안에서 구현: `sigmoid(x) = 1 / (1 + exp(-x))`
- LSTM 대신 GRU를 선택한 이유: 게이트가 더 적고 (2개 vs. 3개) scalar autograd에서 따라가기 쉬우면서, 같은 게이팅 원리를 가르침
- 은닉 상태 차원은 처리 가능하도록 작게 유지 (`n_hidden=32`)
- 마지막에 직접 비교 표를 포함: 아키텍처, 최종 loss, 기울기 norm 비율 (첫 번째 vs. 마지막 타임스텝), 샘플 출력

**역사적 맥락 참고:** 이 스크립트는 "이전" 그림을 제공하기 위해 존재함. Karpathy의 [makemore](https://github.com/karpathy/makemore) 시리즈는 여러 노트북에 걸쳐 이 진행을 다룸. 여기서는 두 모델이 하나의 파일에 있어서 비교가 즉각적이고 피할 수 없음.

**하이퍼파라미터:** `n_hidden=32, seq_len=16, lr=0.1 (SGD), steps=3000 per model, ~800 params (RNN), ~800 params (GRU)`

**성공 기준:**

- Vanilla RNN 기울기 norm 비율 (마지막/첫 번째 타임스텝): < 0.01 (소실을 시연)
- GRU 기울기 norm 비율: 0.1–10.0 (안정성을 시연)
- GRU 최종 loss < vanilla RNN 최종 loss
- GRU에서 생성된 이름이 vanilla RNN보다 높은 품질
- 런타임: M-series Mac에서 < 9분

**예상 복잡도:** ~350-400줄. 두 모델 구현 + 기울기 분석 + 비교 추론.

---

### `microtokenizer.py` — Byte-Pair Encoding

> _"텍스트가 숫자가 되는 방법 — 모든 LLM 안에 숨어 있는 압축 알고리즘."_

**가르치는 것:**

- 토큰화가 중요한 이유 (어휘 효율성, 서브워드 표현)
- BPE merge 알고리즘: 반복적 쌍 빈도 계산과 병합
- 인코딩: 학습된 merge의 탐욕적 좌→우 적용
- 디코딩: token ID에서 바이트 시퀀스로의 단순 조회
- 어휘 크기와 시퀀스 길이의 관계

**알고리즘 개요:**

```
1. 바이트 수준 어휘로 시작 (256개 기본 토큰)
2. 코퍼스에서 모든 인접 토큰 쌍을 카운트
3. 가장 빈번한 쌍을 새 토큰으로 병합
4. N번의 merge를 반복 (N이 어휘 크기를 제어)
5. 인코딩: 새 텍스트에 merge를 탐욕적으로 적용
6. 디코딩: token ID를 바이트 문자열로 매핑
```

**데이터셋:** Karpathy의 makemore에서 가져온 `names.txt` (~32K 이름, urllib로 자동 다운로드)

**핵심 구현 사항:**

- merge 우선순위 테이블 유지 (정렬된 merge 목록)
- 인코딩은 새 텍스트에서 빈도 순이 아닌 우선순위 순으로 merge를 적용
- UTF-8을 제대로 처리: 기본 어휘는 문자가 아닌 바이트 (0-255)
- 토큰화 전/후 압축 비율을 보여줌

**성공 기준:**

- 왕복 정확성: 모든 테스트 입력에 대해 `decode(encode(text)) == text`
- 압축 비율: 바이트 수준 인코딩 대비 토큰 수 1.5배 이상 감소
- 런타임: M-series Mac에서 < 2분

**예상 복잡도:** ~150-200줄. 직관적인 알고리즘, 주된 도전은 깔끔한 encode/decode 대칭.

---

### `microembedding.py` — 대조 임베딩 학습

> _"의미가 기하학이 되는 방법 — 거리가 유사도와 같은 벡터를 학습함."_

**가르치는 것:**

- 학습된 임베딩이 bag-of-words와 TF-IDF보다 의미적 태스크에서 우수한 이유
- InfoNCE / NT-Xent loss를 사용한 대조 학습
- 양성 및 음성 쌍 구성
- 대조 목적 함수에서의 temperature scaling
- 학습된 거리 메트릭으로서의 cosine similarity
- 임베딩 공간이 의미적으로 조직되는 방식

**알고리즘 개요:**

```
1. 간단한 인코더 정의 (bag-of-character-ngrams → linear projection)
2. 학습 쌍 구성:
   - 양성: 같은 문서의 증강 버전 (예: 문자 드롭아웃)
   - 음성: 배치 내 다른 문서들
3. 모든 쌍에 대해 임베딩 계산
4. InfoNCE loss 적용: 양성의 유사도 최대화, 음성의 유사도 최소화
5. SGD/Adam으로 학습
6. 추론: 새 문자열을 임베딩하고 최근접 이웃 찾기
```

**데이터셋:** `names.txt` — 비슷한 발음의 이름이 함께 클러스터링되는 공간에 이름을 임베딩

**핵심 구현 사항:**

- 입력 표현으로 character n-gram 특징 사용 (학습된 tokenizer 의존성 없음)
- 간단한 linear 인코더 (행렬 곱 + 정규화), 딥 네트워크 불필요
- 양성 쌍 생성을 위한 랜덤 문자 삭제/교환 증강
- Cosine similarity 수동 계산: `dot(a,b) / (||a|| * ||b||)`
- 추론 시 최근접 이웃 검색을 시연

**성공 기준:**

- 최근접 이웃이 낮은 편집 거리를 가짐 (예: "Anna" → "Anne", "Anna" → "Zachary" 아님)
- 학습 후 양성 쌍 간 cosine similarity > 0.8
- 학습 후 랜덤 쌍 간 cosine similarity < 0.3
- 런타임: M-series Mac에서 < 5분

**예상 복잡도:** ~200-250줄. loss 함수가 핵심; 인코더는 간단하게 유지 가능.

---

### `microrag.py` — Retrieval-Augmented Generation

> _"검색이 생성을 보강하는 방법 — 실제로 작동하는 가장 단순한 시스템."_

**가르치는 것:**

- RAG 아키텍처: 검색 후 생성
- 문서 검색을 위한 TF-IDF 또는 BM25 스코어링
- 검색된 컨텍스트가 생성 모델의 입력에 주입되는 방법
- 파라메트릭 지식(모델 가중치)과 비파라메트릭 지식(검색된 문서) 사이의 근본적인 트레이드오프
- RAG가 환각을 줄이는 이유

**알고리즘 개요:**

```
1. 문서 인덱스 구축:
   - 문서를 용어로 토큰화
   - TF-IDF (또는 BM25) 점수 계산
   - 역색인으로 저장
2. 쿼리 시점:
   - 모든 문서를 쿼리에 대해 스코어링
   - 상위 k개 문서 검색
3. 검색된 컨텍스트를 쿼리와 연결
4. 보강된 입력을 작은 학습된 언어 모델에 입력
5. 쿼리와 검색된 컨텍스트 모두에 조건화된 출력을 생성
```

**데이터셋:** 100개의 합성 사실 문단 (도시, 국가, 기본 사실). 스크립트 내에서 프로그래밍적으로 생성 — 다운로드 불필요. 검색 품질을 육안으로 검증할 수 있을 만큼 단순함.

**핵심 구현 사항:**

- BM25를 처음부터 구현: term frequency 포화, 문서 길이 정규화, IDF 가중치
- 언어 모델은 연결된 입력(`embed(query) + embed(retrieved_context)`)을 사용하는 **문자 수준 MLP**임. bigram 모델은 검색된 컨텍스트에 조건화할 수 없어서 RAG의 핵심 메커니즘을 시연하는 데 실패함.
- 시연: 검색 유무에 따른 같은 쿼리, 향상된 정확도를 보여줌
- 검색과 생성 컴포넌트 모두 같은 파일에 구현되어야 함

**설계 결정:** MLP는 연결된 쿼리 + 검색된 컨텍스트를 입력으로 받아서, 모델이 실제로 검색된 정보를 사용할 수 있게 함. 이것이 RAG를 의미 있게 시연하는 최소 아키텍처임. 초점은 검색 메커니즘과 컨텍스트 주입에 있지, 생성 모델의 정교함에 있지 않음.

**성공 기준:**

- 검색: BM25가 테스트 쿼리의 80% 이상에 대해 관련 문서를 반환
- 검색 포함 생성이 검색 미포함보다 측정 가능하게 더 나은 출력을 생산
- 런타임: M-series Mac에서 < 6분

**예상 복잡도:** ~350-400줄. 두 서브시스템(BM25 + MLP)이 있어서 기초 스크립트 중 가장 복잡함.

---

### `microdiffusion.py` — Denoising Diffusion

> _"노이즈에서 이미지가 나타나는 방법 — Stable Diffusion 뒤의 알고리즘, 2D로."_

**가르치는 것:**

- 순방향 과정: 데이터에 가우시안 노이즈를 점진적으로 추가
- 역방향 과정: 노이즈를 예측하고 제거하는 것을 학습
- 노이즈 스케줄 (linear beta schedule)
- 디노이징 목적: 추가된 노이즈를 예측
- 샘플링: 순수 노이즈에서 데이터로 반복 디노이징

**알고리즘 개요:**

```
1. 작은 2D 데이터셋 정의 (예: 나선형 또는 Swiss roll에서 샘플링된 점들)
2. 순방향 과정: T 타임스텝에서 linear schedule로 노이즈 추가
3. 작은 MLP를 학습시켜 (noisy_data, timestep)이 주어졌을 때 노이즈를 예측
4. 샘플링: 랜덤 노이즈에서 시작, T 스텝 동안 반복 디노이징
5. 시각화: 생성된 2D 점들 (또는 통계)을 출력
```

**데이터셋:** 합성 — 2D 포인트 클라우드 (나선형, 동심원, Swiss roll). 프로그래밍적으로 생성, 다운로드 불필요.

**핵심 구현 사항:**

- 2D 데이터는 모델을 작게 유지(~1000 params MLP)하면서 모든 알고리즘 구조를 보존
- Linear 노이즈 스케줄: `beta_t`를 `beta_1`에서 `beta_T`까지 선형 보간
- 임의 타임스텝에서 효율적 노이징을 위해 `alpha_bar_t`를 사전 계산
- 디노이징 네트워크는 `[x_noisy, t_embedding]`을 입력으로 받음
- Sinusoidal encoding으로 타임스텝 임베딩 (수동 구현)
- 출력: 생성된 점들의 통계 (평균, 분산, 분포 형태)를 stdout에 출력

**2D-이미지 매핑:** 알고리즘은 이미지 diffusion (Stable Diffusion, DALL-E)과 동일함. 2D 좌표가 픽셀 값에 매핑되고, 1000-param MLP가 10억-param U-Net에 매핑되며, (x,y)에 대한 가우시안 노이즈가 RGB에 대한 가우시안 노이즈에 매핑됨. 핵심 통찰 — 노이즈를 예측하는 법을 학습한 다음, 반복적으로 디노이징 — 은 어떤 차원에서나 동일함. 주석이 이 매핑을 명시적으로 밝혀야 함.

**성공 기준:**

- 생성된 포인트 클라우드 통계 (평균, 분산)가 학습 분포와 20% 이내로 일치
- 육안 검사: 생성된 나선형/Swiss roll이 대상 형태로 인식 가능
- 런타임: M-series Mac에서 < 5분

**예상 복잡도:** ~250-300줄. 2D 단순화로 이미지 라이브러리 없이 처리 가능.

---

### `microvae.py` — Variational Autoencoder

> _"데이터의 압축된 생성적 표현을 학습하는 방법 — reparameterization trick 해체."_

**가르치는 것:**

- 비지도 학습을 위한 인코더-디코더 아키텍처
- Reparameterization trick: 샘플링을 통한 역전파
- ELBO loss: 재구성 loss + KL divergence 정규화
- 잠재 공간 보간과 생성
- VAE가 흐릿하지만 다양한 출력을 생성하는 이유 (GAN과 대비)

**알고리즘 개요:**

```
1. 작은 데이터셋 정의 (2D 점들 또는 작은 이산 시퀀스)
2. 인코더: 입력 → 잠재 분포의 (mean, log_variance) 매핑
3. Reparameterize: z = mean + exp(0.5 * log_var) * epsilon, 여기서 epsilon ~ N(0,1)
4. 디코더: z → 재구성된 입력 매핑
5. Loss = reconstruction_loss + beta * KL(q(z|x) || p(z))
6. Adam으로 학습
7. 추론: z ~ N(0,1) 샘플링, 디코드하여 새 데이터 생성
```

**데이터셋:** 합성 2D 분포 또는 같은 names 데이터셋 (문자 수준 VAE).

**핵심 구현 사항:**

- Reparameterization trick이 명시적이어야 함 — 이것이 교육적 핵심
- 가우시안에 대한 KL divergence는 닫힌 형태의 해가 있음: `0.5 * sum(1 + log_var - mean^2 - exp(log_var))`
- 두 데이터 포인트 간 잠재 공간 보간을 시연
- 재구성/정규화 트레이드오프를 보여주기 위한 Beta-VAE 가중치

**성공 기준:**

- 학습에 걸쳐 재구성 loss가 감소 (ELBO 개선)
- KL divergence가 양수이고 유한 (0으로 붕괴하거나 폭발하지 않음)
- 잠재 보간이 데이터 포인트 간 매끄러운 전이를 생성
- z ~ N(0,1)에서 생성된 샘플이 학습 분포와 유사
- 런타임: M-series Mac에서 < 4분

**예상 복잡도:** ~200-250줄. 개념적으로 우아함; 까다로운 부분은 reparameterization trick을 코드에서 결정적으로 명확하게 만드는 것.

---

## 02 — 정렬 & 학습 기법

### `microlora.py` — Low-Rank Adaptation

> _"파라미터의 1%만 업데이트하여 모델을 파인튜닝하는 방법 — 효율적 적응 뒤의 수학."_

**가르치는 것:**

- 전체 파인튜닝이 비싼 이유 (모든 파라미터, 모든 기울기, 모든 옵티마이저 상태)
- Low-rank 분해: `W_new = W_frozen + A @ B` 여기서 A와 B는 작음
- Low rank가 작동하는 이유 (파인튜닝 중 가중치 업데이트는 경험적으로 low-rank)
- 기본 가중치 동결 vs. 어댑터 가중치 학습
- Rank를 하이퍼파라미터로: 용량 vs. 효율 트레이드오프

**알고리즘 개요:**

```
1. 데이터셋 A에서 기본 모델 학습 (microgpt 아키텍처 재사용)
2. 모든 기본 모델 파라미터 동결
3. 선택된 가중치 행렬에 low-rank 어댑터 추가: A (d×r)와 B (r×d), r << d
4. 순전파: output = W_frozen @ x + A @ B @ x
5. A와 B만 기울기를 받음
6. 데이터셋 B (다른 분포)로 학습
7. 보여줌: 적응된 모델이 A를 잊지 않으면서 B에서 수행함
```

**데이터셋:** `names.txt`의 두 분할 — 예: A-M으로 시작하는 이름은 기본, N-Z는 적응 대상. 또는: 영어 이름을 기본으로, 다른 이름 목록을 적응으로.

**핵심 구현 사항:**

- 기본 모델이 먼저 학습됨 (microgpt 루프 재사용)
- 어댑터 행렬 초기화: A ~ N(0, σ), B = 0 (초기 적응이 영)
- 명시적 기울기 동결: 기본 `Value` 노드는 backward 후 `.grad`가 0으로 리셋
- 비교: LoRA 유무에 따른 학습 가능 파라미터 수
- 기본 및 적응 분포 모두에서 생성 품질을 보여줌

**성공 기준:**

- 기본 모델이 데이터셋 A에서 수렴 (loss < 2.5)
- LoRA 적응 모델이 A에서의 치명적 망각 없이 데이터셋 B에서 개선
- LoRA를 사용한 학습 가능 파라미터 수 < 전체 모델 파라미터의 10%
- 런타임: M-series Mac에서 < 7분 (기본: 50% 수렴까지 3분 + LoRA: 2분 + 추론: 1분)

**예상 복잡도:** ~350-400줄. 기본 학습과 LoRA 적응 단계를 모두 포함.

---

### `microdpo.py` — Direct Preference Optimization

> _"별도의 보상 모델을 학습하지 않고 인간 선호도에 맞게 모델을 정렬하는 방법."_

**가르치는 것:**

- 선호도 학습 문제: (prompt, chosen, rejected)가 주어졌을 때, 모델이 "chosen"을 선호하게 만드는 것
- DPO가 RLHF를 단순화하는 이유: 최적 정책은 보상과 닫힌 형태의 관계를 가짐
- DPO loss 함수: log-probability 비율에 대한 대조 목적 함수
- 참조 모델의 역할 (KL 앵커)
- Beta 파라미터: 선호도가 기본 분포를 얼마나 강하게 오버라이드하는지

**알고리즘 개요:**

```
1. 텍스트 코퍼스에서 기본/참조 모델 학습
2. 선호도 쌍 생성: (prompt, chosen_completion, rejected_completion)
3. 정책과 참조 모델 모두에서 chosen과 rejected의 log-probability 계산
4. DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
5. 정책 모델만 업데이트
6. 보여줌: 생성이 선호 완성 쪽으로 이동
```

**데이터셋:** `names.txt`에서 파생된 합성 선호도 쌍 — 예: 특정 음성적 특성을 가진 이름을 선호, 또는 짧은 이름보다 긴 이름을 선호. 선호 신호는 육안으로 검증할 수 있을 만큼 단순해야 함.

**핵심 구현 사항:**

- 참조 모델은 기본 모델 파라미터의 동결된 복사본
- Log-probability 계산은 전체 시퀀스 스코어링 필요 (토큰별 log-prob의 합)
- Sigmoid과 log-ratio 수학은 수치적으로 안정적이어야 함
- Beta가 정렬 강도를 제어 — 다른 값으로 시연
- 보여줌: 학습에 따라 정책과 참조 간 KL divergence가 증가

**성공 기준:**

- 학습에 걸쳐 DPO loss가 감소
- 정책 모델이 기각된 것보다 선호 완성을 더 빈번하게 생성
- 정책과 참조 간 KL divergence가 학습에 따라 증가 (beta로 제어)
- 런타임: M-series Mac에서 < 7분 (사전학습: 3분 + DPO: 3분 + 추론: 1분)

**예상 복잡도:** ~350-400줄. 두 모델 복사본(참조 + 정책)과 선호도 쌍 구성이 필요.

---

### `microppo.py` — Proximal Policy Optimization for RLHF

> _"전체 RLHF 루프: 보상 모델, policy gradient, KL 페널티 — 하나의 파일에 전부."_

**가르치는 것:**

- RLHF 파이프라인: 사전학습 → 보상 모델 → 정책 최적화
- 선호도 쌍으로부터의 보상 모델 학습
- Clipped surrogate objective를 사용한 policy gradient (PPO)
- 정책이 참조에서 너무 멀어지는 것을 방지하는 KL 페널티
- 분산 감소를 위한 value function baseline
- 이것이 DPO보다 어려운 이유 (그리고 언제 여전히 이것을 원할 것인지)

**알고리즘 개요:**

```
1. 기본 언어 모델 학습 (사전학습 단계)
2. 선호도 쌍으로 보상 모델 학습:
   - 입력: (prompt, completion) → scalar 보상 점수
   - Pairwise ranking loss로 학습
3. PPO 루프:
   a. 현재 정책에서 완성 생성
   b. 보상 모델로 완성 점수 매기기
   c. 이점 계산 (보상 - value baseline)
   d. Clipped surrogate objective로 정책 업데이트
   e. Value function 업데이트
   f. 참조 정책에 대한 KL 페널티 적용
4. 보여줌: 보상 신호에 따라 생성 품질 향상
```

**데이터셋:** 비교 가능성을 위해 `microdpo.py`와 같은 합성 선호도 설정.

**핵심 구현 사항:**

- **Hybrid autograd 접근:** 정책 모델은 scalar autograd (`Value` class)를 사용. PPO 기울기가 정책을 통해 흘러야 하기 때문. 보상 모델과 value function은 수동 기울기 계산이 포함된 일반 float 배열 사용 — PPO 루프 전에 별도로 학습되므로, autograd 오버헤드가 불필요함. 런타임 제약 내에서 전체 RLHF 알고리즘을 보존함.
- 정책: scalar autograd, ~1,000 params (`n_embd=8, n_head=2, n_layer=1`)
- 보상 모델: 일반 float MLP, ~500 params, pairwise ranking loss로 학습
- Value function: 일반 float linear, ~200 params
- PPO clipping: `min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)`
- KL 페널티: 시퀀스별 `KL(policy || reference)` 명시적 계산
- 이점 추정: 단순 `reward - value_baseline` (처리 가능하도록 GAE 없음)
- 학습: 100 PPO 스텝, batch_size=4, seq_len=8
- 컬렉션에서 가장 복잡한 스크립트 — 그에 맞게 시간 배정할 것

```python
# IMPLEMENTATION NOTE: The reward model and value function use plain floats (not
# autograd Value objects) for runtime tractability. The policy model uses scalar
# autograd because PPO gradients must flow through the policy's generation process.
# Production RLHF (InstructGPT, ChatGPT) vectorizes all three models on GPUs;
# we split the approach to stay within pure-Python runtime constraints while
# preserving the complete PPO algorithm.
```

**성공 기준:**

- 보상 모델 정확도: 홀드아웃 선호도 쌍에서 > 70%
- 정책 생성이 학습에 걸쳐 선호 완성 쪽으로 이동
- PPO loss가 감소; KL divergence가 증가 (페널티 계수로 제어)
- 런타임: M-series Mac에서 < 7분

**예상 복잡도:** ~550-600줄. 가장 야심찬 스크립트; 세 개의 상호작용하는 모델 + RL 루프.

---

### `micromoe.py` — Mixture of Experts

> _"연산을 스케일링하지 않고 모델 용량을 스케일링하는 방법 — sparse routing 실전."_

**가르치는 것:**

- MoE 개념: 다수의 expert 네트워크, 입력당 일부만 활성화
- Router/gating 네트워크: 토큰이 expert에 할당되는 방법
- Top-k expert 선택과 expert 출력의 소프트 결합
- Load balancing loss: 순진한 라우팅이 왜 하나의 expert만 사용하게 붕괴하는지
- MoE가 스케일에서 매력적인 용량 vs. 연산 트레이드오프

**알고리즘 개요:**

```
1. N개의 작은 expert MLP 정의 (각각 동일 아키텍처, 다른 가중치)
2. Router 정의: 입력을 N개 expert 점수로 매핑하는 linear layer
3. 각 입력 토큰에 대해:
   a. Router가 N개 expert에 대한 점수를 생성
   b. Top-k expert 선택 (보통 k=2)
   c. 선택된 expert 출력의 가중 합 계산
4. Expert 붕괴를 방지하기 위한 load balancing 보조 loss 추가
5. 언어 모델링 목적 + 보조 loss로 학습
6. 보여줌: expert 활용 통계와 expert별 특화
```

**데이터셋:** `names.txt` 또는 expert에게 특화할 충분한 신호를 주기 위한 약간 더 큰 텍스트 코퍼스.

**핵심 구현 사항:**

- **Hybrid autograd 접근:** Router는 scalar autograd (`Value` class)를 사용. 라우팅 결정이 MoE의 핵심 메커니즘이므로 기울기가 gating function을 통해 흘러야 함. Expert MLP는 런타임 처리 가능성을 위해 수동 기울기 계산이 포함된 일반 float 배열 사용.
- 4개 expert, top-2 라우팅 (의미 있는 load balancing 역학 보존)
- Router는 softmax를 사용한 간단한 linear layer (autograd `Value` 객체)
- Expert는 각 ~200 params의 2층 MLP (일반 float)
- Load balancing loss: expert 할당 빈도의 분산 최소화
- 학습 스텝마다 expert 활용도를 추적하고 출력
- Expert 특화를 시연: 어떤 expert가 어떤 입력 패턴에 활성화되는지

```python
# IMPLEMENTATION NOTE: Experts use plain floats (not autograd Value objects) for
# runtime tractability. The router uses scalar autograd because routing decisions
# are the core MoE mechanism — gradients must flow through the gating function.
# Production MoE frameworks (Mixtral, Switch Transformer) vectorize everything;
# we split the approach to stay within pure-Python runtime constraints.
```

**성공 기준:**

- 4개 expert 모두 토큰 할당의 >10%를 받음 (expert 붕괴 없음)
- 학습에 걸쳐 load balancing loss가 감소
- 다른 expert가 다른 입력 패턴에서 측정 가능한 특화를 보임
- 런타임: M-series Mac에서 < 7분

**예상 복잡도:** ~350-400줄. 라우팅 로직과 보조 loss가 핵심; 각 expert는 단순한 MLP.

---

## 03 — 시스템 & 추론

### `microattention.py` — Attention 변형 모음

> _"중요한 모든 attention 메커니즘, 하나의 파일에서 나란히 구현."_

**가르치는 것:**

- Vanilla scaled dot-product attention
- Multi-head attention (병렬 head, 연결, 프로젝션)
- Grouped-query attention (GQA): query head 간 공유 KV head
- Multi-query attention (MQA): 모든 query head에 대해 단일 KV head
- Sliding window attention: 로컬 컨텍스트 윈도우
- 각 변형이 메모리, 연산, 품질을 어떻게 트레이드오프하는지

**알고리즘 개요:**

```
각 attention 변형에 대해:
1. 순전파 구현
2. FLOP 및 메모리 사용량을 분석적으로 계산
3. 같은 입력 시퀀스에서 실행
4. 출력: 출력 값, FLOP 수, 메모리 풋프린트
5. 모든 변형을 요약 테이블로 비교
```

**데이터셋:** 학습 없음. 메커니즘을 시연하기 위해 랜덤 입력 텐서 (`Value` 또는 일반 float의 리스트 of 리스트) 사용.

**핵심 구현 사항:**

- 이것은 주로 **순전파 비교**이지, 학습 스크립트가 아님 (초점이 아키텍처 비교에 있으므로, 학습+추론 규칙의 예외로 정당화됨)
- 각 변형은 자기 완결적 함수
- 마지막에 비교 테이블 출력: 변형, FLOP, 메모리, 출력 유사도
- GQA와 MQA는 MHA 기본의 수정으로 구현, 차이를 명시적으로 만듦

**성공 기준:**

- 모든 변형이 유효한 attention 출력을 생성 (NaN 없음, 오버플로우 없음)
- MQA와 GQA 출력이 MHA에 근접 (cosine similarity > 0.95)
- 출력된 비교 테이블이 올바른 FLOP/메모리 트레이드오프를 보여줌
- 런타임: M-series Mac에서 < 1분

**예상 복잡도:** ~250-300줄. 하나의 큰 구현이 아닌 여러 작은 구현.

---

### `microkv.py` — KV-Cache 메커니즘

> _"LLM 추론이 왜 메모리 바운드인지 — 그리고 KV cache가 정확히 어떻게 작동하는지."_

**가르치는 것:**

- 각 생성 스텝에서 순진하게 attention을 실행하면 왜 O(n²)의 중복인지
- KV cache: 이전 위치의 key/value 프로젝션을 저장하고 재사용
- 메모리 성장: 캐시 크기가 시퀀스 길이, 레이어, head에 따라 어떻게 스케일링되는지
- Paged attention 직관: 왜 메모리 단편화가 스케일에서 중요한지
- Prefill vs. decode 단계

**알고리즘 개요:**

```
1. KV cache 없이 attention 구현 (매 스텝 모든 것을 재계산)
2. KV cache 포함 attention 구현 (점진적, append-only)
3. 둘 다 같은 자기회귀 생성 태스크에서 실행
4. 비교: 각 스텝의 연산 수, 메모리 사용량
5. Paged 할당 시뮬레이션: 고정 크기 블록, 포인터 테이블
```

**데이터셋:** 사전학습된 작은 모델 (인라인 학습 또는 작은 가중치 하드코딩)을 생성에 사용.

**핵심 구현 사항:**

- 나란히 놓은 구현이 중복을 명백하게 만듦
- 각 생성 스텝에서 곱셈 연산을 세고 출력
- 메모리 성장 곡선을 보여줌: 시퀀스 위치에 따른 캐시 크기
- Paged attention 섹션은 개념적/시뮬레이션 — 전체 메모리 관리자 없이 할당 전략을 시연

**성공 기준:**

- 캐시 포함과 미포함이 동일한 출력을 생성
- 연산 수: 캐시 미포함은 O(n²)로 성장, 캐시 포함은 O(n)로 성장
- 메모리 성장 곡선이 출력되고 선형 스케일링을 보여줌
- 런타임: M-series Mac에서 < 4분

**예상 복잡도:** ~200-250줄. 비교 구조가 교육 도구.

---

### `microquant.py` — 가중치 양자화

> _"최소한의 품질 손실로 모델을 4배 축소하는 방법 — INT8과 INT4 뒤의 수학."_

**가르치는 것:**

- 양자화가 작동하는 이유: 신경망 가중치는 대략적으로 정규분포
- Absmax 양자화: 정수 범위에 맞게 스케일링
- Zero-point 양자화: 비대칭 범위
- Per-channel vs. per-tensor 양자화 세분성
- 추론을 위한 역양자화
- 품질 저하 측정

**알고리즘 개요:**

```
1. 작은 모델을 수렴까지 학습 (microgpt 아키텍처 재사용)
2. 가중치를 INT8로 양자화:
   a. 스케일 팩터 계산: max(abs(weights)) / 127
   b. 양자화: round(weight / scale)
   c. 정수 + 스케일 팩터로 저장
3. 가중치를 INT4로 양자화 (같은 과정, 범위 [-8, 7])
4. 역양자화하고 각 버전으로 추론 실행
5. 비교: 모델 크기, 생성 품질, 토큰별 loss
```

**데이터셋:** `names.txt` — 기본 모델을 학습한 다음, 양자화하고 비교.

**핵심 구현 사항:**

- 양자화된 가중치를 float가 아닌 Python 정수로 표현 — 이것이 핵심
- 실제 메모리 절약을 보여줌: `float32 (4 bytes) → int8 (1 byte) → int4 (0.5 bytes)`
- 각 양자화 수준별 perplexity/loss를 계산하고 출력
- Per-channel vs. per-tensor 시연: 가중치 행렬의 각 행을 별도로 양자화 vs. 전체 행렬
- 왕복 테스트: 양자화 → 역양자화 → 원본과 비교

**성공 기준:**

- INT8 양자화 모델 loss가 float32 기준의 10% 이내
- INT4 양자화 모델 loss가 float32 기준의 25% 이내
- Per-channel 양자화가 per-tensor 양자화보다 우수
- 출력 테이블이 모델 크기 감소를 보여줌: float32 → INT8 (4x) → INT4 (8x)
- 런타임: M-series Mac에서 < 6분

**예상 복잡도:** ~300-350줄. 기본 학습 + 양자화 + 비교 평가를 포함.

---

### `microflash.py` — Flash Attention (알고리즘 시뮬레이션)

> _"Flash Attention이 빠른 이유 — tiling과 online softmax 트릭, 순수 Python으로 시뮬레이션."_

**가르치는 것:**

- 표준 attention의 메모리 병목: 전체 N×N attention 행렬의 구체화
- Tiled 연산: "빠른 메모리"에 맞는 블록 단위로 attention 처리
- Online softmax: 모든 점수를 저장하지 않고 softmax를 점진적으로 계산
- IO 복잡도 논증: 더 적은 메모리 읽기가 더 적은 FLOP보다 왜 더 중요한지
- 메모리 접근 패턴: Flash Attention이 더 빠른 진짜 이유

**알고리즘 개요:**

```
1. 표준 attention 구현 (전체 N×N 행렬 구체화)
2. Flash Attention 구현:
   a. Q, K, V를 크기 B 블록으로 타일링
   b. 각 Q 블록에 대해:
      - 각 K,V 블록에 대해:
        - 부분 attention 점수 계산
        - Online 알고리즘으로 running softmax 업데이트
        - 가중 값 누적
   c. 최종 출력 리스케일
3. 검증: 출력이 표준 attention과 일치 (부동소수점 허용 오차 내)
4. 비교: 최대 "메모리" 사용량 (시뮬레이션), 메모리 읽기/쓰기 횟수
```

**데이터셋:** 학습 없음. 메커니즘을 시연하기 위한 설정 가능한 크기의 랜덤 행렬.

**핵심 구현 사항:**

- **이것은 알고리즘 시뮬레이션이지, 성능 최적화가 아님.** 순수 Python은 표준 attention보다 느릴 것임. 요점은 Flash Attention이 _무엇을_ 하는지 보여주는 것이지, 빠른 것이 아님.
- Online softmax가 핵심 통찰: 블록 간 running `max`와 `sum`을 유지
- 시뮬레이션된 메모리 사용량을 추적하고 출력: 표준 (O(N²)) vs. flash (O(N))
- 설정 가능한 블록 크기 B로 tiling 세분성이 메모리에 미치는 영향을 보여줌
- 수치 검증: 출력이 1e-6 이내로 일치하는지 assert

**성공 기준:**

- Flash attention 출력이 1e-6 허용 오차 내에서 표준 attention과 일치
- 시뮬레이션된 최대 메모리: 표준 O(N²) vs. flash O(N)이 명확히 보여짐
- 런타임: M-series Mac에서 < 2분

**예상 복잡도:** ~300-350줄. Online softmax가 핵심; tiling 부기와 비교 출력이 예상보다 더 많은 줄을 추가.

---

### `microbeam.py` — 디코딩 전략

> _"탐욕을 넘어서: beam search, top-k, top-p, speculative decoding을 하나의 파일에."_

**가르치는 것:**

- Greedy decoding: 각 스텝에서 argmax를 취하기 (그리고 왜 차선인지)
- Temperature sampling: 무작위성 제어
- Top-k sampling: 가장 가능성 높은 k개 토큰으로 절단
- Top-p (nucleus) sampling: 누적 확률 임계값으로 절단
- Beam search: 상위 B개 후보를 유지하고 완성 시퀀스를 스코어링
- Speculative decoding: 작은 모델로 초안을 작성하고 큰 모델로 검증

**알고리즘 개요:**

```
1. 인라인으로 두 언어 모델 학습 (microgpt의 autograd 패턴 재사용):
   - 큰 "target" 모델: n_embd=16, n_layer=1 (~4,200 params)
   - 작은 "draft" 모델: n_embd=8, n_layer=1 (~1,300 params)
2. 각 디코딩 전략을 별도 함수로 구현
3. 각 전략으로 같은 프롬프트에서 생성
4. 출력: 생성된 텍스트, 총 log-probability, 생성 속도 (시뮬레이션)
5. Speculative decoding:
   a. 작은 "draft" 모델이 k개 토큰을 탐욕적으로 생성
   b. 큰 "target" 모델이 k개 토큰을 모두 병렬로 스코어링
   c. 일치하는 토큰 수락, 첫 불일치에서 거부하고 재샘플링
```

**데이터셋:** `names.txt` — 같은 학습된 모델에서 다른 전략으로 이름을 생성.

**핵심 구현 사항:**

- 모든 전략이 같은 기반 모델에서 동작하여, 비교가 공정
- Beam search는 B개의 독립적 KV cache 유지 (또는 재계산) 필요
- Speculative decoding은 두 모델 크기 사용 (다른 `n_embd` / `n_layer` 설정)
- 비교 테이블 출력: 전략, 출력, log-prob, tokens/step

**성공 기준:**

- 모든 디코딩 전략이 유효한 토큰 시퀀스를 생성
- Beam search가 greedy보다 높은 log-probability 시퀀스를 생성
- Top-p와 top-k가 greedy보다 더 다양한 출력을 생성 (고유 이름 수로 측정)
- Speculative decoding이 평균적으로 draft 토큰의 50% 이상 수락
- 출력 비교 테이블: 전략, 출력, log-prob, tokens/step
- 런타임: M-series Mac에서 < 7분

**예상 복잡도:** ~450-500줄. 많은 작은 구현 + speculative decoding 두 모델 설정 + 인라인 학습.

---

## 구현 우선순위 & 순서

스크립트는 의존성을 관리하고 공유 autograd/모델 패턴을 검증하기 위해 이 순서로 구축됨. 표준 autograd 인터페이스(`docs/autograd-interface.md`)는 Phase 2 전에 확정됨.

| 단계        | 스크립트                                          | 근거                                                                                                                                                                                       |
| ----------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Phase 1** | `microtokenizer.py`, `microembedding.py`          | Autograd 의존성 없음, 독립형 알고리즘                                                                                                                                                      |
| **Phase 2** | `microgpt.py`, `micrornn.py`, `microattention.py` | 표준 autograd `Value` class 패턴 확립. microgpt가 참조 구현; micrornn이 `sigmoid`으로 확장. microattention은 순전파만 (autograd 없음).                                                      |
| **Phase 3** | `microrag.py`, `microlora.py`                     | microrag는 문자 수준 MLP 사용 (더 가벼운 autograd 의존성). microlora는 microgpt의 학습 패턴에 직접 기반.                                                                                   |
| **Phase 4** | `microdiffusion.py`, `microvae.py`                | 독립적 알고리즘, 다른 모델 계열. Phase 3과 병렬화 가능.                                                                                                                                    |
| **Phase 5** | `microdpo.py`, `microppo.py`                      | Phase 2의 안정적 autograd 패턴 필요. microppo는 hybrid autograd 사용 (정책: Value class, 보상/value: 일반 float).                                                                          |
| **Phase 6** | `microquant.py`, `microkv.py`, `microflash.py`    | 시스템 스크립트, Phase 3-5와 독립적으로 구축 가능                                                                                                                                          |
| **Phase 7** | `microbeam.py`, `micromoe.py`                     | microbeam은 인라인으로 두 모델 학습 (Phase 2 패턴 의존). micromoe는 hybrid autograd 사용 (router: Value class, expert: 일반 float).                                                        |

### 의존성 참고사항

- **Phase 2가 크리티컬 패스.** 이후 6개 스크립트가 재구현하는 autograd `Value` class를 확립함. 표준 인터페이스가 여기서 검증되어야 함.
- **Phase 3과 4는 병렬 실행 가능** — 교차 의존성 없음.
- **Phase 5 스크립트** (DPO, PPO)는 Phase 2에서 autograd 패턴이 안정적이어야 함. 기본 모델 학습 루프를 내부적으로 재구현.
- **Hybrid autograd 스크립트** (microppo, micromoe)는 `Value` 객체와 일반 float를 혼합 사용. 표준 인터페이스는 autograd 부분에 여전히 적용됨.

## 품질 체크리스트

전체 품질 체크리스트(실행, 주석, 가독성, 물류)는 `CONTRIBUTING.md`를 참조할 것. 해당 문서가 PR 리뷰 기준의 유일한 권위 있는 참조임.

---

_각 스크립트는 하나의 증명임. 알고리즘은 생각보다 단순함._
