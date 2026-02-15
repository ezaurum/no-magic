# 표준 Autograd 인터페이스

이 문서는 모든 스크립트가 구현해야 하는 scalar autograd `Value` class 인터페이스를 정의함. 이 인터페이스는 scalar automatic differentiation을 사용하는 9개 스크립트 전반의 일관성을 보장하면서, 스크립트별 확장은 허용함.

## 이 문서가 존재하는 이유

각 스크립트는 자기 완결적(공유 import 없음)이므로, `Value` class는 필요한 모든 스크립트에서 재구현됨. 표준 인터페이스가 없으면:

- 구현이 서로 달라짐 (하나는 `sigmoid`을 지원하는데, 다른 하나는 안 됨)
- 수치 안정성 패턴이 일관되지 않게 적용됨
- 이후 스크립트에서 autograd 섹션을 건너뛴 독자가 스크립트별 차이를 놓침

이 스펙은 **최소** 인터페이스를 정의함. 스크립트는 추가 연산을 넣을 수 있음 (아래의 autograd callout 패턴으로 문서화할 것).

---

## 필수 연산

모든 `Value` class는 다음 연산을 지원해야 함:

### 산술 연산

| 연산      | Python 메서드             | 비고                                              |
| --------- | ------------------------- | ------------------------------------------------- |
| 덧셈      | `__add__`, `__radd__`     | `Value + Value`, `Value + float`, `float + Value` |
| 곱셈      | `__mul__`, `__rmul__`     | `Value * Value`, `Value * float`, `float * Value` |
| 부정      | `__neg__`                 | `-Value` (`self * -1`로 구현)                     |
| 뺄셈      | `__sub__`, `__rsub__`     | `__add__`와 `__neg__`를 통해 구현                 |
| 나눗셈    | `__truediv__`             | `__mul__`과 `__pow__(-1)`을 통해 구현             |
| 거듭제곱  | `__pow__`                 | `Value ** int` 또는 `Value ** float`              |

### 활성화 함수

| 함수     | 시그니처                 | Backward                                        |
| -------- | ------------------------ | ----------------------------------------------- |
| `tanh()` | `self.tanh() -> Value`   | `grad * (1 - out**2)`                           |
| `exp()`  | `self.exp() -> Value`    | `grad * out`                                    |
| `relu()` | `self.relu() -> Value`   | `grad * (1 if self.data > 0 else 0)`            |
| `log()`  | `self.log() -> Value`    | `grad / self.data` (`self.data >= 1e-10`으로 클램프) |

### Backward Pass

```python
def backward(self):
    """역전파 자동미분으로 기울기를 계산함 (위상 정렬)."""
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

### 기울기 관리

```python
# 각 학습 스텝 전에, 모든 기울기를 0으로 초기화
for p in params:
    p.grad = 0.0
```

---

## 스크립트별 확장

기본 세트를 넘어서 추가 연산이 필요한 스크립트는:

1. `Value` class에 확장을 구현할 것
2. 아래의 autograd callout 패턴으로 문서화할 것

### 스크립트별 알려진 확장

| 스크립트         | 추가 연산               | 필요한 이유                                      |
| ---------------- | ----------------------- | ------------------------------------------------ |
| `microgpt.py`    | (기본 외 없음)          | 표준 인터페이스의 참조 구현                      |
| `micrornn.py`    | `sigmoid()`             | GRU 게이팅: `z_t = sigmoid(...)`                 |
| `microlora.py`   | (기본 외 없음)          | 기본 세트 사용                                   |
| `microdpo.py`    | (기본 외 없음)          | `log()`는 기본 세트에 포함됨                     |
| `microppo.py`    | `clip()`                | PPO ratio clipping                               |
| `micromoe.py`    | (router만)              | Router는 기본 세트 사용; expert는 일반 float     |
| `microkv.py`     | (기본 외 없음)          | 학습 전용 간결한 Value class                     |
| `microquant.py`  | (기본 외 없음)          | 학습에는 autograd; 양자화에는 float 사용         |
| `microbeam.py`   | (기본 외 없음)          | 학습에는 autograd; 디코딩에는 float 사용         |

### Autograd Callout 패턴

`Value` class를 사용하는 모든 스크립트는 class 정의 직후에 이 블록을 포함해야 함:

```python
# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# This Value class follows the canonical interface (see docs/autograd-interface.md)
# with the following additions/modifications:
# - sigmoid(): Required for GRU gating (z_t and r_t computations)
# - [list any other additions]
# Base operations (add, mul, tanh, exp, relu, pow, backward) are identical
# to the canonical spec.
```

추가 사항이 없는 스크립트의 경우:

```python
# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.
```

---

## 수치 안정성 패턴

이 패턴들은 사용하는 모든 스크립트에서 **필수**임. 각 패턴에는 수치적 근거를 설명하는 주석이 반드시 있어야 함.

### Stable Softmax

```python
def softmax(logits):
    # 수치 안정 softmax: exp 전에 max를 빼서 오버플로우를 방지함.
    # softmax는 이동 불변: 임의의 c에 대해 softmax(x) = softmax(x - c).
    # 이것 없으면, x > 709일 때 exp(x)가 오버플로우됨 (Python math.exp 한계).
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]
```

### Clipped Log-Probability

```python
def safe_log(x):
    # log(0)을 방지함. log(0)은 -inf를 반환하고 기울기 계산을 깨뜨림.
    # 1e-10으로 클램프하면 log(1e-10) ≈ -23이 되어, 유한하고
    # 0에 가까운 확률에 대한 기울기 정보를 보존함.
    #
    # 중요: 기울기가 계산 그래프를 통해 역전파되도록 x를 자식으로 하는
    # log 노드를 직접 생성함. Value(clamped).log()를 사용하면
    # 연결이 끊긴 노드가 생겨서, 기울기 경로가 완전히 단절됨.
    clamped = max(x.data, 1e-10)
    return Value(math.log(clamped), (x,), (1.0 / clamped,))
```

### Adam Epsilon

```python
def adam_step(param, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    # eps는 v (2차 모멘트)가 0에 가까울 때 0으로 나누는 것을 방지함.
    # 표준 값: 1e-8 (PyTorch/TensorFlow 기본값과 동일).
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    param.data -= lr * m / (v ** 0.5 + eps)
```

### KL Divergence Clamping (microvae 전용)

```python
def kl_divergence(mean, log_var):
    # log_var를 [-5, 5]로 클램프하여 exp(log_var) 폭발을 방지함.
    # exp(5) = 148 (합리적인 분산); exp(10) = 22,026 (KL이 폭발함).
    clamped_lv = max(min(log_var.data, 5.0), -5.0)
    return Value(0.5) * (Value(1.0) + Value(clamped_lv) - mean * mean - Value(clamped_lv).exp())
```

---

## 테스트 벡터

`Value` class 구현이 올바른 기울기를 생성하는지 검증하는 데 사용할 것:

### 테스트 1: 단순 체인

```python
a = Value(2.0)
b = Value(3.0)
c = a * b + b  # c = 2*3 + 3 = 9
c.backward()
assert a.grad == 3.0   # dc/da = b = 3
assert b.grad == 3.0   # dc/db = a + 1 = 3
```

### 테스트 2: Tanh 기울기

```python
x = Value(0.5)
y = x.tanh()  # y = tanh(0.5) = 0.4621
y.backward()
# dy/dx = 1 - tanh(0.5)^2 = 1 - 0.2135 = 0.7865
assert abs(x.grad - 0.7865) < 0.001
```

### 테스트 3: 재사용 (기울기 누적)

```python
a = Value(2.0)
b = a + a  # a가 두 번 사용됨
b.backward()
assert a.grad == 2.0  # db/da = 1 + 1 = 2
```

### 테스트 4: Softmax 안정성

```python
logits = [Value(1000.0), Value(1001.0), Value(1002.0)]
probs = softmax(logits)
# 오버플로우되면 안 됨. 예상값: ~[0.09, 0.24, 0.67]
assert all(0 < p.data < 1 for p in probs)
assert abs(sum(p.data for p in probs) - 1.0) < 1e-6
```

---

## 구현 참고사항

- **Python 객체 오버헤드:** 각 `Value`는 `.data` (float), `.grad` (float), `._backward` (클로저), `._prev` (set)를 저장함. 대략적인 메모리: Value당 ~100 바이트.
- **파라미터 예산:** 7분 런타임 제약은 스크립트당 총 모델 파라미터를 ~5,000개 Value 객체로 사실상 제한함. 이를 초과하는 스크립트(microppo, micromoe)는 hybrid autograd를 사용함.
- **결정론:** `set` 순회 순서는 해시 랜덤화로 인해 Python 세션마다 달라짐. 엄격한 재현성이 필요하면, 스크립트 헤더에 `PYTHONHASHSEED=0`을 기록할 것. 교육 목적에서 실행 간 미세한 수치 차이는 허용됨.
- **기울기 초기화:** 모든 `backward()` 호출 전에 반드시 수행해야 함. 이것을 빼먹는 게 가장 흔한 autograd 버그 — 안 하면 학습 스텝마다 기울기가 누적됨.
