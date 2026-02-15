"""
Mixture of Experts (MoE): router 네트워크가 각 토큰을 specialist MLP의 부분집합으로
라우팅하는 법을 학습하여, 연산량을 비례적으로 늘리지 않고 모델 용량을 확장함.
"""
# Reference: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
# Mixture-of-Experts Layer" (2017). https://arxiv.org/abs/1701.06538
# Also: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with
# Simple and Efficient Sparsity" (2021). https://arxiv.org/abs/2101.03961
# Architecture reuses the microgpt embedding/LM-head pattern (Radford et al., 2019) with
# the transformer block replaced by a routed MoE layer.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# 모델 아키텍처
N_EMBD = 8           # embedding 차원 -- MoE가 넓은 표현이 아닌 expert 수를 통해
                      # 용량을 추가하므로 microgpt(16)보다 작음
N_EXPERTS = 4        # expert MLP 수
TOP_K = 2            # 토큰당 선택되는 expert 수 -- top-2가 표준 MoE 선택임;
                      # top-1(Switch Transformer)이 더 단순하지만 routing 오류에 덜 강건함
EXPERT_HIDDEN = 16   # 각 expert MLP 내의 hidden 차원 (N_EMBD에서 2배 확장)
BLOCK_SIZE = 12      # context window 길이

# 학습 파라미터
LEARNING_RATE = 0.01
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
NUM_STEPS = 800
AUX_LOSS_COEFF = 0.1  # load balancing auxiliary loss의 가중치 -- language modeling 품질과
                       # 균등한 expert 활용 사이의 트레이드오프를 제어함.
                       # 너무 낮으면: router가 1-2개 expert로 붕괴됨. 너무 높으면: 균일한
                       # routing을 강제하여 전문화를 방해함. 0.1이 표준 시작점임.

# Data
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 참고: 총 ~2,000개 파라미터. 프로덕션 MoE 모델(Mixtral-8x7B, Switch-C)은 수백 개
# expert에 걸쳐 수십억 개 파라미터를 가짐. routing 알고리즘은 동일하고 expert 크기와
# 수만 다름. 4개 expert, top-2 설정으로 전체 동작을 포착함: router 학습, load balancing,
# expert 전문화, sparse activation.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """학습 코퍼스를 다운로드하고 파싱함."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    with open(filename, "r") as f:
        docs = [line.strip() for line in f if line.strip()]

    return docs


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """reverse-mode 자동 미분을 지원하는 scalar 값.

    모든 forward 연산은 local derivative(dout/dinput)를 기록함. backward()는
    역 위상 정렬 순서로 computation graph를 재생하며, chain rule을 통해 gradient를
    누적함: dLoss/dx = 경로들의 합(경로를 따른 local gradient들의 곱).
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()) -> None:
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self) -> Value: return self * -1
    def __radd__(self, other: float) -> Value: return self + other
    def __sub__(self, other: Value | float) -> Value: return self + (-other)
    def __rsub__(self, other: float) -> Value: return other + (-self)
    def __rmul__(self, other: float) -> Value: return self * other
    def __truediv__(self, other: Value | float) -> Value: return self * (other ** -1)
    def __rtruediv__(self, other: float) -> Value: return other * (self ** -1)

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self) -> Value:
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self) -> None:
        """computation graph의 위상 정렬을 통한 reverse-mode autodiff."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# 이 Value 클래스는 표준 인터페이스(docs/autograd-interface.md 참조)를 따르며
# 추가 사항 없음. Router는 자동 미분을 위해 Value 객체를 사용함.
# Expert MLP는 수동 gradient 계산으로 일반 float를 사용함.
# See docs/autograd-interface.md for the full canonical interface.

# 구현 참고: Expert는 런타임 효율을 위해 일반 float를 사용함(autograd Value 객체가
# 아님). Router는 routing 결정이 MoE의 핵심 메커니즘이므로 scalar autograd를
# 사용함 -- gradient가 gating function을 통해 흘러야 함.
# 프로덕션 MoE 프레임워크(Mixtral, Switch Transformer)는 모든 것을 벡터화함.
# 순수 Python 런타임 제약 내에서 유지하기 위해 접근 방식을 분리함.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """가중치 행렬 초기화 ~ N(0, std). 표준편차 0.08은 이 tiny model에 맞게 경험적으로
    튜닝됨. 더 큰 모델은 Xavier/Glorot 스케일링(std = 1/sqrt(d_in))을 사용함."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def make_float_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[float]]:
    """expert MLP용 일반 float 가중치 행렬 초기화."""
    return [[random.gauss(0, std) for _ in range(ncols)] for _ in range(nrows)]


def init_expert_weights() -> list[dict[str, list[list[float]]]]:
    """4개의 독립적인 expert MLP를 각각의 가중치로 초기화함.

    각 expert는 2-layer MLP: input(N_EMBD) -> hidden(EXPERT_HIDDEN) -> output(N_EMBD).
    Expert들이 서로 다른 랜덤 가중치로 시작하므로 학습 중 다른 입력 패턴에
    전문화될 수 있음. 모든 expert가 동일하게 시작하면 router가 하나를 다른 것보다
    선호할 이유가 없고, 대칭 깨짐이 전적으로 노이즈에 의존하게 됨.
    """
    experts = []
    for _ in range(N_EXPERTS):
        # w1: [EXPERT_HIDDEN, N_EMBD] 입력을 hidden 차원으로 투영
        # w2: [N_EMBD, EXPERT_HIDDEN] hidden을 embedding 차원으로 다시 투영
        expert = {
            'w1': make_float_matrix(EXPERT_HIDDEN, N_EMBD),
            'w2': make_float_matrix(N_EMBD, EXPERT_HIDDEN),
        }
        experts.append(expert)
    return experts


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """행렬-벡터 곱: y = W @ x. W의 shape이 [n_out, n_in]이고 x의 shape이
    [n_in]이면, 출력 y의 shape은 [n_out]이며 y[i] = sum_j W[i,j] * x[j]."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """안정적인 softmax: overflow 방지를 위해 exp 전에 max를 뺌.
    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))"""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMS 정규화: x / sqrt(mean(x^2) + eps).
    LayerNorm보다 단순함(mean centering 없음, 학습 가능한 affine 없음). LLaMA, Gemma에서 사용됨."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """수치 안정성을 위한 clipped log. log(0) = -inf를 방지하여 gradient 전파가
    깨지는 것을 막음. prob을 child로 가지는 노드를 수동으로 구성하여
    gradient가 computation graph를 통해 흐르게 함(clamping으로 끊기지 않음)."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === EXPERT FORWARD PASS (PLAIN FLOATS) ===

def expert_forward_float(x: list[float], weights: dict[str, list[list[float]]]) -> list[float]:
    """단일 expert MLP forward pass: x -> hidden(ReLU) -> output. 모두 일반 float.

    수식: hidden = ReLU(W1 @ x), output = W2 @ hidden
    표준 2-layer MLP임. 각 expert가 다른 입력-출력 매핑을 학습하므로
    MoE layer는 단일 MLP의 4배 용량을 가짐 -- 하지만 토큰당 2개 expert만
    활성화하여 연산은 단일 expert의 2배(4배가 아님)로 유지됨.
    """
    w1 = weights['w1']
    w2 = weights['w2']

    # Hidden layer: ReLU activation과 함께 W1 @ x
    hidden = [sum(w1[i][j] * x[j] for j in range(len(x))) for i in range(len(w1))]
    hidden = [max(0.0, h) for h in hidden]  # ReLU

    # Output layer: W2 @ hidden (embedding 차원으로 다시 투영)
    output = [sum(w2[i][j] * hidden[j] for j in range(len(hidden))) for i in range(len(w2))]
    return output


def expert_backward_float(
    x: list[float],
    weights: dict[str, list[list[float]]],
    output_grads: list[float],
    lr: float,
) -> None:
    """단일 expert MLP의 수동 gradient 계산 및 가중치 업데이트.

    Expert의 출력이 Value 객체로 래핑되고 router 점수와 곱해질 때,
    backward()가 해당 Value 래퍼에 .grad를 설정함. 이 gradient를 output_grads로
    추출하고 expert의 일반 float layer를 통해 수동으로 전파함.

    Expert MLP를 통한 chain rule:
        d(loss)/d(w2[i][j]) = output_grads[i] * hidden[j]
        d(loss)/d(hidden[j]) = sum_i(output_grads[i] * w2[i][j]) * relu_grad(pre_relu[j])
        d(loss)/d(w1[i][j]) = hidden_grads[i] * x[j]

    표준 backpropagation -- Value 클래스가 자동화하는 것과 동일한 알고리즘이지만,
    여기서는 일반 float expert 가중치에 대해 수동으로 수행함.
    """
    w1 = weights['w1']
    w2 = weights['w2']

    # --- 중간 activation을 얻기 위해 forward pass 재계산 ---
    pre_relu = [sum(w1[i][j] * x[j] for j in range(len(x))) for i in range(len(w1))]
    hidden = [max(0.0, h) for h in pre_relu]

    # --- W2를 통한 backward: output = W2 @ hidden ---
    # d(loss)/d(w2[i][j]) = output_grads[i] * hidden[j]
    for i in range(len(w2)):
        for j in range(len(w2[i])):
            w2[i][j] -= lr * output_grads[i] * hidden[j]

    # --- ReLU를 통해 hidden layer로 backward ---
    # d(loss)/d(hidden[j]) = sum_i(output_grads[i] * w2[i][j])
    # d(loss)/d(pre_relu[j]) = d(loss)/d(hidden[j]) * (1 if pre_relu[j] > 0 else 0)
    hidden_grads = [0.0] * len(w1)
    for j in range(len(hidden)):
        for i in range(len(w2)):
            hidden_grads[j] += output_grads[i] * w2[i][j]
        # ReLU gradient: pre-activation이 양수였으면 통과, 아니면 0
        if pre_relu[j] <= 0:
            hidden_grads[j] = 0.0

    # --- W1을 통한 backward: pre_relu = W1 @ x ---
    # d(loss)/d(w1[i][j]) = hidden_grads[i] * x[j]
    for i in range(len(w1)):
        for j in range(len(w1[i])):
            w1[i][j] -= lr * hidden_grads[i] * x[j]


# === MOE FORWARD PASS ===

def moe_forward(
    token_id: int,
    pos_id: int,
    params: dict[str, list[list[Value]]],
    expert_weights: list[dict[str, list[list[float]]]],
) -> tuple[list[Value], list[Value], list[int], list[float]]:
    """단일 토큰에 대한 MoE 모델 forward pass.

    아키텍처:
        1. 토큰 임베딩 (token + position embedding)
        2. RMSNorm
        3. Router: N_EXPERTS 점수로의 linear projection, softmax로 확률 변환
        4. Router 확률 기준 top-K expert 선택
        5. 선택된 expert 실행(일반 float), 출력을 Value 객체로 래핑
        6. Router 점수를 사용한 expert 출력의 가중 합
        7. LM head: 어휘 logit으로 투영

    반환값:
        logits: 어휘 크기의 logit 벡터 (Value 객체)
        router_probs: 전체 router 확률 분포 (Value 객체, aux loss용)
        selected_experts: 선택된 top-K expert의 인덱스
        x_float: expert 입력의 일반 float (backward pass를 위해 캐시됨)
    """
    # --- 토큰 임베딩 ---
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # --- Router: 어떤 expert가 이 토큰을 처리할지 결정 ---
    # Router는 softmax가 뒤따르는 단순한 linear layer임. 토큰의 표현을
    # expert에 대한 확률 분포로 매핑함.
    # 수식: router_probs = softmax(W_router @ x)
    # 이 softmax를 통해 Value 클래스로 gradient가 흐르므로, router가
    # 어떤 expert가 어떤 토큰에 최적인지 학습함.
    router_logits = linear(x, params['w_router'])
    router_probs = softmax(router_logits)

    # --- Top-K expert 선택 ---
    # Router 확률이 가장 높은 K개 expert를 선택함.
    # Sparse activation이 MoE의 핵심 특성: N_EXPERTS만큼의 파라미터를 가지지만
    # 토큰당 TOP_K개의 expert forward pass만 계산함. 이것이 MoE가
    # "연산 확장 없이 용량을 확장"하는 방법임.
    scored = [(router_probs[i].data, i) for i in range(N_EXPERTS)]
    scored.sort(reverse=True)
    selected_experts = [idx for _, idx in scored[:TOP_K]]

    # --- 선택된 expert 점수 재정규화 ---
    # Top-K 선택 후, 확률의 합이 1이 되도록 재정규화함.
    # 선택되지 않은 expert가 얼마나 확률 질량을 가졌든 가중 조합이
    # 적절하게 스케일링되도록 보장함.
    selected_scores = [router_probs[i] for i in selected_experts]
    score_sum = sum(s.data for s in selected_scores)
    if score_sum > 1e-10:
        norm_scores = [s / score_sum for s in selected_scores]
    else:
        norm_scores = [s for s in selected_scores]

    # --- Expert 계산 (일반 float) ---
    # Value 기반 표현을 expert MLP용 일반 float로 추출함.
    # Expert가 출력을 계산한 후, 결과를 다시 Value 객체로 래핑하고
    # router 점수와 곱함 -- 이것이 autograd router와 일반 float expert 사이의
    # gradient bridge를 생성함.
    x_float = [v.data for v in x]

    # --- Expert 출력의 가중 조합 ---
    # 수식: output = sum_i(score_i * expert_i(x)) (i는 선택된 expert)
    # 선택된 각 expert가 동일한 입력을 독립적으로 처리한 뒤, (재정규화된)
    # router 확률을 가중치로 사용하여 출력을 블렌딩함.
    combined = [Value(0.0)] * N_EMBD
    for k_idx, expert_idx in enumerate(selected_experts):
        expert_out = expert_forward_float(x_float, expert_weights[expert_idx])

        # Expert 출력을 Value 객체로 래핑하여 router 점수(Value)와의 곱셈이
        # computation graph 노드를 생성하게 함. backward() 후, Value 래퍼가
        # d(loss)/d(expert_output)를 누적하며, 이를 사용하여 expert 가중치를
        # 수동으로 업데이트함.
        for j in range(N_EMBD):
            expert_val = Value(expert_out[j])
            combined[j] = combined[j] + norm_scores[k_idx] * expert_val

    # --- LM head: 어휘로 투영 ---
    logits = linear(combined, params['lm_head'])
    return logits, router_probs, selected_experts, x_float


# === LOAD BALANCING AUXILIARY LOSS ===

def compute_aux_loss(
    expert_assignment_counts: list[int],
    router_prob_sums: list[float],
    total_tokens: int,
) -> Value:
    """현재 학습 스텝의 load balancing auxiliary loss를 계산함.

    이 loss 없이는 router가 붕괴됨: 학습 초기에 약간 더 낮은 loss를 내는
    1-2개 expert에게만 모든 토큰을 보내는 법을 학습함. 사용되지 않는 expert는
    gradient를 받지 못하고 랜덤 초기화 상태로 남음 -- "expert collapse" 또는
    "rich get richer"라 불리는 양의 피드백 루프임.

    Auxiliary loss는 두 양의 곱을 통해 불균등한 분배에 페널티를 줌:
        f_i = expert i에 할당된 토큰 비율 (이진 할당 지시자)
        P_i = expert i의 평균 router 확률 (부드러운 연속 신호)

    수식: L_aux = N_EXPERTS * sum_i(f_i * P_i)

    왜 f_i * P_i의 곱인가? Expert i가 많은 토큰을 받고(높은 f_i) 높은 평균
    확률을 가지면(높은 P_i) 곱이 커지고 loss가 이를 페널티함. 최솟값은
    모든 expert에서 f_i = P_i = 1/N(균등 분포)일 때 발생함.

    N_EXPERTS 스케일링 팩터는 서로 다른 expert 수에서 loss 크기가 대략
    비슷하게 하여, AUX_LOSS_COEFF가 expert 수에 특화된 튜닝이 필요 없게 함.
    """
    if total_tokens == 0:
        return Value(0.0)

    aux = Value(0.0)
    for i in range(N_EXPERTS):
        # f_i: expert i로 라우팅된 토큰 비율
        f_i = expert_assignment_counts[i] / total_tokens
        # P_i: 모든 토큰에 대한 expert i의 평균 router 확률
        p_i = router_prob_sums[i] / total_tokens
        # f_i * P_i의 곱이 자주 선택되면서 높은 router 확률을 받는
        # expert에 페널티를 줌
        aux = aux + Value(f_i * p_i)

    return aux * N_EXPERTS


# === TRAINING ===

if __name__ == "__main__":
    start_time = time.time()

    # -- 데이터 로드 및 준비 --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} documents")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # === Initialize Model Parameters ===

    params: dict[str, list[list[Value]]] = {}

    # Token과 position embedding (Value 객체)
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    # Router: embedding 공간에서 expert 점수로의 linear projection
    # Shape: [N_EXPERTS, N_EMBD] -- expert당 하나의 점수
    params['w_router'] = make_matrix(N_EXPERTS, N_EMBD)

    # LM head: MoE 출력을 어휘 logit으로 투영
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    # Expert MLP (일반 float -- autograd로 추적되지 않음)
    expert_weights = init_expert_weights()

    # -- 아키텍처 요약을 위한 파라미터 수 계산 --
    router_params = N_EXPERTS * N_EMBD
    expert_params_each = EXPERT_HIDDEN * N_EMBD + N_EMBD * EXPERT_HIDDEN  # w1 + w2
    expert_params_total = expert_params_each * N_EXPERTS
    embd_params = VOCAB_SIZE * N_EMBD + BLOCK_SIZE * N_EMBD + VOCAB_SIZE * N_EMBD  # wte + wpe + lm_head

    print(f"\n=== MoE Model Architecture ===")
    print(f"Router parameters: {router_params} (Value class autograd)")
    print(f"Expert parameters: {expert_params_each} x {N_EXPERTS} experts = {expert_params_total} (plain floats)")
    print(f"Embedding parameters: {embd_params} (Value class autograd)")
    print(f"Total parameters: {router_params + expert_params_total + embd_params:,}")

    # -- Adam optimizer를 위한 autograd 파라미터 수집 --
    autograd_param_list: list[Value] = []
    for matrix in params.values():
        for row in matrix:
            autograd_param_list.extend(row)

    m_state = [0.0] * len(autograd_param_list)
    v_state = [0.0] * len(autograd_param_list)

    # -- Expert 활용도 추적 --
    # 붕괴를 감지하기 위해 모든 토큰에서 어떤 expert가 선택되는지 추적함.
    # 건강한 MoE는 토큰을 대략 균등하게 분배함. 붕괴는 1-2개 expert가
    # 대다수의 할당을 받는 것을 의미함.
    cumulative_expert_counts = [0] * N_EXPERTS
    utilization_report_interval = 200

    # 보고용 smoothed loss -- 각 스텝이 단일 문서로 학습하므로 개별 스텝 loss가
    # 노이즈가 많음. 지수 이동 평균(alpha=0.05)이 ~20 스텝에 걸쳐 평활화하여
    # 학습 진행의 더 정확한 그림을 제공함.
    smooth_lm_loss = 3.3  # 예상 시작 loss 근처로 초기화
    smooth_alpha = 0.05

    # === Training Loop ===
    print(f"\n=== Training ===")

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        # Auxiliary loss 계산을 위한 스텝별 추적
        step_expert_counts = [0] * N_EXPERTS
        step_router_prob_sums = [0.0] * N_EXPERTS
        step_token_count = 0

        losses: list[Value] = []
        for pos in range(seq_len):
            input_token = tokens[pos]
            target_token = tokens[pos + 1]

            logits, router_probs, selected_experts, x_float = moe_forward(
                input_token, pos, params, expert_weights,
            )

            # Expert 활용도 추적
            for eidx in selected_experts:
                step_expert_counts[eidx] += 1
                cumulative_expert_counts[eidx] += 1

            # Auxiliary loss를 위한 router 확률 추적
            for i in range(N_EXPERTS):
                step_router_prob_sums[i] += router_probs[i].data
            step_token_count += 1

            # Cross-entropy loss: -log P(target)
            probs = softmax(logits)
            loss_t = -safe_log(probs[target_token])
            losses.append(loss_t)

        # -- 총 loss 계산: LM loss + auxiliary load balancing loss --
        lm_loss = (1.0 / seq_len) * sum(losses)
        aux_loss = compute_aux_loss(step_expert_counts, step_router_prob_sums, step_token_count)
        total_loss = lm_loss + AUX_LOSS_COEFF * aux_loss

        # -- Autograd graph를 통한 backward pass --
        total_loss.backward()

        # -- Adam으로 autograd 파라미터(embedding, router, LM head) 업데이트 --
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, param in enumerate(autograd_param_list):
            m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * param.grad
            v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * param.grad ** 2
            m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            param.grad = 0.0

        # -- 수동 gradient 계산을 통한 expert 가중치 업데이트 --
        # Autograd는 expert 가중치(일반 float)에 도달할 수 없음. 표준
        # cross-entropy gradient를 사용하여 각 토큰에 대해
        # d(loss)/d(expert_output)를 분석적으로 계산한 뒤,
        # 각 expert MLP를 통해 수동으로 역전파함.
        #
        # Gradient 경로: loss -> softmax -> logits -> lm_head -> combined -> score * expert_out
        # Cross-entropy gradient d(-log softmax(z)[t])/d(z[i]) = softmax(z)[i] - 1{i==t}는
        # 잘 알려져 있으며 비용이 큰 finite difference를 피함.
        expert_lr = lr_t * 0.5  # expert용 낮은 LR -- SGD가 Adam보다 노이즈가 많음

        # LM head를 스텝당 한 번 float로 캐시 (position에 걸쳐 상수)
        lm_head_float = [[v.data for v in row] for row in params['lm_head']]

        for pos in range(seq_len):
            input_token = tokens[pos]
            target_token = tokens[pos + 1]

            # Router 결정과 expert 입력을 복구하기 위해 부분 forward 재실행
            tok_emb = params['wte'][input_token]
            pos_emb = params['wpe'][pos]
            x = [t + p for t, p in zip(tok_emb, pos_emb)]
            x = rmsnorm(x)

            router_logits = linear(x, params['w_router'])
            router_probs_re = softmax(router_logits)
            scored = [(router_probs_re[i].data, i) for i in range(N_EXPERTS)]
            scored.sort(reverse=True)
            selected = [idx for _, idx in scored[:TOP_K]]

            selected_scores_data = [router_probs_re[i].data for i in selected]
            score_sum = sum(selected_scores_data)
            if score_sum > 1e-10:
                norm_score_data = [s / score_sum for s in selected_scores_data]
            else:
                norm_score_data = selected_scores_data

            x_float_re = [v.data for v in x]

            # 각 선택된 expert를 실행하고 결합된 출력을 계산
            expert_outputs: dict[int, list[float]] = {}
            for eidx in selected:
                expert_outputs[eidx] = expert_forward_float(x_float_re, expert_weights[eidx])

            combined_float = [0.0] * N_EMBD
            for k_idx, eidx in enumerate(selected):
                for j in range(N_EMBD):
                    combined_float[j] += norm_score_data[k_idx] * expert_outputs[eidx][j]

            # Cross-entropy gradient를 위한 softmax(logits) 계산
            logits_float = [
                sum(lm_head_float[i][j] * combined_float[j] for j in range(N_EMBD))
                for i in range(VOCAB_SIZE)
            ]
            max_logit = max(logits_float)
            exp_logits = [math.exp(l - max_logit) for l in logits_float]
            sum_exp = sum(exp_logits)
            probs_float = [e / sum_exp for e in exp_logits]

            # d(loss)/d(logits[i]) = softmax(logits)[i] - 1{i == target}
            d_logits = [probs_float[i] - (1.0 if i == target_token else 0.0)
                        for i in range(VOCAB_SIZE)]

            # d(loss)/d(combined[j]) = sum_i d(loss)/d(logits[i]) * lm_head[i][j]
            d_combined = [0.0] * N_EMBD
            for j in range(N_EMBD):
                for i in range(VOCAB_SIZE):
                    d_combined[j] += d_logits[i] * lm_head_float[i][j]
                d_combined[j] /= seq_len  # 평균 LM loss에 맞게 스케일링

            # 가중 조합을 통한 chain: d(loss)/d(expert_out) = d_combined * score
            for k_idx, eidx in enumerate(selected):
                d_expert_out = [d_combined[j] * norm_score_data[k_idx] for j in range(N_EMBD)]
                expert_backward_float(
                    x_float_re, expert_weights[eidx], d_expert_out, expert_lr,
                )

        # -- Smoothed loss 업데이트 --
        smooth_lm_loss = smooth_alpha * lm_loss.data + (1 - smooth_alpha) * smooth_lm_loss

        # -- 로깅 --
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | lm_loss: {lm_loss.data:.4f} "
                  f"(smooth: {smooth_lm_loss:.4f}) | aux_loss: {aux_loss.data:.4f} | "
                  f"total: {total_loss.data:.4f}")

        if (step + 1) % utilization_report_interval == 0:
            total_assignments = sum(cumulative_expert_counts)
            if total_assignments > 0:
                pcts = [100 * c / total_assignments for c in cumulative_expert_counts]
                pct_str = " ".join(f"E{i}={pcts[i]:.0f}%" for i in range(N_EXPERTS))
                print(f"  step {step + 1:>4}: {pct_str}")

    elapsed_train = time.time() - start_time
    print(f"\nTraining complete. Smoothed LM loss: {smooth_lm_loss:.4f}")
    print(f"Training time: {elapsed_train:.1f}s")

    # === Expert Analysis ===
    print(f"\n=== Expert Analysis ===")
    total_assignments = sum(cumulative_expert_counts)
    print("Final expert utilization:")
    all_above_threshold = True
    for i in range(N_EXPERTS):
        pct = 100 * cumulative_expert_counts[i] / total_assignments if total_assignments > 0 else 0
        print(f"  Expert {i}: {pct:.1f}% of tokens")
        if pct < 10.0:
            all_above_threshold = False

    if all_above_threshold:
        print("\nAll experts receive >10% of tokens (no collapse)")
    else:
        print("\nWARNING: Expert collapse detected — some experts below 10%")

    print(f"Load balancing loss: {aux_loss.data:.4f}")

    # === Generation ===
    TEMPERATURE = 0.7
    NUM_SAMPLES = 15

    print(f"\n=== Generation ===")
    print(f"Generating {NUM_SAMPLES} samples (temperature={TEMPERATURE}):\n")

    for sample_idx in range(NUM_SAMPLES):
        token_id = BOS
        generated: list[str] = []
        experts_used: set[int] = set()

        for pos in range(BLOCK_SIZE):
            # 생성을 위한 forward pass (gradient 추적 불필요하지만
            # 일관성을 위해 동일한 함수를 재사용함)
            logits, router_probs, selected, _ = moe_forward(
                token_id, pos, params, expert_weights,
            )
            experts_used.update(selected)

            # Temperature 스케일링된 sampling
            scaled_logits = [logit / TEMPERATURE for logit in logits]
            probs = softmax(scaled_logits)

            token_id = random.choices(
                range(VOCAB_SIZE), weights=[p.data for p in probs]
            )[0]

            if token_id == BOS:
                break
            generated.append(unique_chars[token_id])

        name = ''.join(generated)
        experts_str = ','.join(str(e) for e in sorted(experts_used))
        print(f"  {sample_idx + 1:>2}. {name} (experts used: {experts_str})")

    elapsed_total = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_total:.1f}s")
