"""
전체 RLHF 루프: language model을 pretrain하고, human preference로 reward model을 학습한 뒤,
Proximal Policy Optimization으로 policy를 최적화함 -- 전부 하나의 파일에서, 처음부터 구현함.
"""
# Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
# https://arxiv.org/abs/1707.06347
# Also: Ouyang et al., "Training language models to follow instructions with human
# feedback" (InstructGPT, 2022). https://arxiv.org/abs/2203.02155
# Architecture reuses the microgpt pattern (Radford et al., 2019) with a smaller model
# (n_embd=8) to accommodate the three-model RLHF pipeline within runtime constraints.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Policy model 아키텍처 -- 3개 모델 파이프라인을 7분 런타임 제한 내에 맞추기 위해
# microgpt보다 작게 설정함. 아키텍처 자체는 동일하고(attention, MLP,
# residual connection), 차원만 줄임.
N_EMBD = 8          # embedding 차원 (microgpt의 16 대비)
N_HEAD = 2          # attention head 수 (4 대비)
N_LAYER = 1         # transformer 블록 수
BLOCK_SIZE = 12     # context window (16 대비) -- 이름이 보통 3-8자이므로 짧게 설정함
HEAD_DIM = N_EMBD // N_HEAD  # head당 4차원

# Pretraining 파라미터
PRETRAIN_LR = 0.01
PRETRAIN_STEPS = 500
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8

# Reward model 파라미터
REWARD_HIDDEN = 32       # hidden layer 너비
REWARD_LR = 0.01         # SGD 학습률
REWARD_STEPS = 400        # 학습 반복 횟수
REWARD_MARGIN = 1.0       # ranking loss margin -- preferred/rejected 간 명확한 분리를 강제함

# PPO 파라미터
PPO_CLIP_EPS = 0.2       # clipping epsilon -- policy ratio가 얼마나 벗어날 수 있는지 제한함
KL_COEFF = 0.5           # KL penalty 계수 -- 일반적인 값(0.01-0.1)보다 높음. 작은 모델과
                         # synthetic reward 환경에서 mode collapse가 잘 발생하기 때문임
PPO_STEPS = 100          # PPO 최적화 스텝 수 (과최적화 방지를 위해 적게 설정함)
BATCH_SIZE = 4           # PPO 스텝당 생성하는 completion 수
MAX_GEN_LEN = 8          # 최대 생성 길이
MIN_GEN_LEN = 2          # 최소 생성 길이 -- 퇴화된 빈 출력에 페널티를 줌
PPO_LR = 0.0005          # pretraining보다 낮게 설정하여 catastrophic forgetting 방지함
VALUE_LR = 0.01          # value function 학습률

# Data 파라미터
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 구현 참고: reward model과 value function은 런타임 효율을 위해 일반 float를 사용함
# (autograd Value 객체가 아님). policy model은 PPO gradient가 policy의 생성 과정을
# 통해 흘러야 하므로 scalar autograd를 사용함.
# 프로덕션 RLHF(InstructGPT, ChatGPT)는 세 모델 모두 GPU에서 벡터화함.
# 순수 Python 런타임 제약 내에서 완전한 PPO 알고리즘을 보존하기 위해
# 이렇게 접근 방식을 분리함.


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

    def clip(self, low: float, high: float) -> Value:
        """값을 [low, high] 범위로 clamp함. 범위 내에 있으면 gradient가 통과하고,
        clamp되면 0임. PPO ratio clipping에 필수: clipped surrogate objective가
        확률 ratio를 clamp하여 치명적으로 큰 policy 업데이트를 방지함.
        ratio가 [1-eps, 1+eps] 범위 내이면 gradient가 정상적으로 흐름.
        clamp되면 gradient가 0이 됨 -- 이것이 PPO의 "proximal" 제약임."""
        clamped = max(low, min(high, self.data))
        grad = 1.0 if low < self.data < high else 0.0
        return Value(clamped, (self,), (grad,))

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
# 다음이 추가됨:
# - clip(): PPO ratio clipping에 필요함 (값을 clamp하고, 범위 내에서 gradient를 통과시킴)
# See docs/autograd-interface.md for the full canonical interface.


# === PARAMETER INITIALIZATION (POLICY MODEL -- VALUE CLASS) ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """가중치 행렬 초기화 ~ N(0, std). 작은 std로 tiny model에서 activation 폭발을
    방지함. Xavier scaling(1/sqrt(d_in))이면 std=0.35가 되어 너무 큼."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_policy_params(vocab_size: int) -> dict[str, list[list[Value]]]:
    """모든 policy model 파라미터 초기화: embedding, attention, MLP, LM head."""
    params: dict[str, list[list[Value]]] = {}

    params['wte'] = make_matrix(vocab_size, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    for layer_idx in range(N_LAYER):
        params[f'layer{layer_idx}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)
        # MLP: 4배 확장 후 축소 (표준 GPT feedforward)
        params[f'layer{layer_idx}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)

    params['lm_head'] = make_matrix(vocab_size, N_EMBD)

    return params


def flatten_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """파라미터 dict에서 모든 Value 객체를 flat 리스트로 수집함."""
    return [p for matrix in params.values() for row in matrix for p in row]


# === CORE OPERATIONS (POLICY MODEL -- VALUE CLASS) ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """행렬-벡터 곱: y = W @ x. Shape: [n_out, n_in] @ [n_in] -> [n_out]."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """안정적인 softmax: overflow 방지를 위해 exp 전에 max를 뺌.
    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))"""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMS 정규화: x / sqrt(mean(x^2) + eps). LayerNorm보다 단순함."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """수치 안정성을 위한 clipped log. log(0) = -inf를 방지함. prob을 child로 가지는
    노드를 수동으로 구성하여 gradient가 graph를 통해 흐르게 함."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === POLICY MODEL FORWARD PASS ===

def policy_forward(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
) -> list[Value]:
    """policy GPT의 단일 토큰 forward pass. 어휘에 대한 logit을 반환함.

    microgpt의 forward pass와 구조적으로 동일하지만 차원이 더 작음
    (n_embd=8, n_head=2). KV cache가 점진적으로 쌓이므로 causal masking이
    암묵적임: position t에서 position 0..t의 key/value만 존재함.
    """
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, params[f'layer{layer_idx}.attn_wq'])
        k = linear(x, params[f'layer{layer_idx}.attn_wk'])
        v = linear(x, params[f'layer{layer_idx}.attn_wv'])

        keys[layer_idx].append(k)
        values[layer_idx].append(v)

        x_attn: list[Value] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM

            q_head = q[hs : hs + HEAD_DIM]
            k_head = [k_t[hs : hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[hs : hs + HEAD_DIM] for v_t in values[layer_idx]]

            # Scaled dot-product attention: score = (q . k) / sqrt(d_head)
            attn_logits = [
                sum(q_head[j] * k_head[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]
            attn_weights = softmax(attn_logits)

            head_output = [
                sum(attn_weights[t] * v_head[t][j] for t in range(len(v_head)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_output)

        x = linear(x_attn, params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        # MLP: 확장, ReLU, 축소
        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === REWARD MODEL (PLAIN FLOATS -- NO AUTOGRAD) ===
# Reward model은 문자 시퀀스를 점수화하는 단순한 MLP임. PPO 루프 이전에
# 독립적으로 학습되므로(pairwise ranking loss와 수동 SGD) autograd 오버헤드가
# 불필요하여 일반 float 배열을 사용함. 프로덕션 reward model은 대형 transformer이지만,
# 이 MLP가 toy 스케일에서 동일한 preference 학습 메커니즘을 구현함.

def init_reward_model(d_in: int) -> dict[str, list[list[float]]]:
    """reward model 초기화: feature 벡터를 scalar로 매핑하는 2-layer MLP."""
    params: dict[str, list[list[float]]] = {}
    # Hidden layer: d_in -> REWARD_HIDDEN
    params['w1'] = [[random.gauss(0, 0.1) for _ in range(d_in)] for _ in range(REWARD_HIDDEN)]
    params['b1'] = [[0.0] for _ in range(REWARD_HIDDEN)]
    # Output layer: REWARD_HIDDEN -> 1
    params['w2'] = [[random.gauss(0, 0.1) for _ in range(REWARD_HIDDEN)]]
    params['b2'] = [[0.0]]
    return params


def reward_forward(features: list[float], params: dict[str, list[list[float]]]) -> float:
    """reward MLP의 forward pass. scalar reward 점수를 반환함.

    아키텍처: input -> linear -> ReLU -> linear -> scalar
    입력은 문자 빈도와 시퀀스 길이를 인코딩하는 feature 벡터임.
    순서 정보는 손실되지만 preference 신호(이름 길이, 문자 분포)를
    최소한의 파라미터로 포착함.
    """
    # ReLU가 있는 hidden layer
    hidden = []
    for i in range(REWARD_HIDDEN):
        h = sum(params['w1'][i][j] * features[j] for j in range(len(features))) + params['b1'][i][0]
        hidden.append(max(0.0, h))  # ReLU

    # Output layer: scalar
    score = sum(params['w2'][0][j] * hidden[j] for j in range(REWARD_HIDDEN)) + params['b2'][0][0]
    return score


def sequence_to_features(token_ids: list[int], vocab_size: int) -> list[float]:
    """토큰 ID 시퀀스를 reward/value model용 feature 벡터로 변환함.

    Feature: 정규화된 문자 카운트(vocab_size 차원) + 정규화된 길이(1 차원).
    길이 feature가 중요함: preference 신호가 이름 길이에 기반하므로(4-7자 선호)
    reward model이 이에 대한 명시적 접근이 필요함. 시퀀스 길이가 아닌 고정 상수로
    카운트를 정규화하여 긴 시퀀스가 더 큰 feature 크기를 갖게 하고,
    문자 feature에서도 길이 신호를 보존함.
    """
    features = [0.0] * (vocab_size + 1)  # 길이 feature를 위해 +1
    for tid in token_ids:
        if 0 <= tid < vocab_size:
            features[tid] += 1.0
    # 문자 카운트를 대략 [0, 1] 범위로 스케일링
    for i in range(vocab_size):
        features[i] /= 10.0
    # 길이 feature: [0, 1]로 정규화 (실용적 최대 이름 길이 ~10자)
    features[vocab_size] = len(token_ids) / 10.0
    return features


def reward_backward(
    features_chosen: list[float],
    features_rejected: list[float],
    params: dict[str, list[list[float]]],
    lr: float,
) -> float:
    """pairwise ranking loss와 수동 SGD의 한 스텝.

    수식: loss = max(0, margin - (reward_chosen - reward_rejected))
    이 hinge loss는 reward model이 chosen 시퀀스를 rejected 시퀀스보다 최소
    `margin`만큼 높게 점수화하도록 밀어줌. InstructGPT에서 reward model 학습에
    사용된 것과 동일한 objective이며, 전체 Bradley-Terry 프레임워크를 단순화한 것임.

    reward model이 일반 float를 사용하므로(autograd Value 객체가 아님)
    gradient를 chain rule로 수동 계산함.
    """
    d_in = len(features_chosen)

    # Chosen에 대한 forward pass
    hidden_c = []
    pre_relu_c = []
    for i in range(REWARD_HIDDEN):
        h = sum(params['w1'][i][j] * features_chosen[j] for j in range(d_in)) + params['b1'][i][0]
        pre_relu_c.append(h)
        hidden_c.append(max(0.0, h))
    score_c = sum(params['w2'][0][j] * hidden_c[j] for j in range(REWARD_HIDDEN)) + params['b2'][0][0]

    # Rejected에 대한 forward pass
    hidden_r = []
    pre_relu_r = []
    for i in range(REWARD_HIDDEN):
        h = sum(params['w1'][i][j] * features_rejected[j] for j in range(d_in)) + params['b1'][i][0]
        pre_relu_r.append(h)
        hidden_r.append(max(0.0, h))
    score_r = sum(params['w2'][0][j] * hidden_r[j] for j in range(REWARD_HIDDEN)) + params['b2'][0][0]

    # Hinge loss: max(0, margin - (score_chosen - score_rejected))
    diff = score_c - score_r
    loss = max(0.0, REWARD_MARGIN - diff)

    if loss <= 0.0:
        # Margin이 만족됨 -- gradient 없음, 업데이트 불필요
        return loss

    # Backward pass: d_loss/d_diff = -1 (active 상태일 때 loss = margin - diff이므로)
    # d_diff/d_score_c = 1, d_diff/d_score_r = -1
    d_score_c = -1.0  # d_loss/d_score_c
    d_score_r = 1.0   # d_loss/d_score_r

    # Output layer를 통한 gradient
    d_hidden_c = [params['w2'][0][j] * d_score_c for j in range(REWARD_HIDDEN)]
    d_hidden_r = [params['w2'][0][j] * d_score_r for j in range(REWARD_HIDDEN)]

    # Output layer 가중치 업데이트
    for j in range(REWARD_HIDDEN):
        params['w2'][0][j] -= lr * (hidden_c[j] * d_score_c + hidden_r[j] * d_score_r)
    params['b2'][0][0] -= lr * (d_score_c + d_score_r)

    # ReLU와 hidden layer를 통한 gradient
    for i in range(REWARD_HIDDEN):
        relu_grad_c = 1.0 if pre_relu_c[i] > 0 else 0.0
        relu_grad_r = 1.0 if pre_relu_r[i] > 0 else 0.0
        d_pre_c = d_hidden_c[i] * relu_grad_c
        d_pre_r = d_hidden_r[i] * relu_grad_r

        for j in range(d_in):
            params['w1'][i][j] -= lr * (d_pre_c * features_chosen[j] + d_pre_r * features_rejected[j])
        params['b1'][i][0] -= lr * (d_pre_c + d_pre_r)

    return loss


def score_completion(token_ids: list[int], vocab_size: int,
                     reward_params: dict[str, list[list[float]]]) -> float:
    """reward model로 completion을 점수화하되, 정규화를 적용함.

    raw reward model 출력은 0 근처에 중심이 있고 스케일이 알 수 없음. 단순한
    정규화를 적용함: 학습 분포 시퀀스에서의 평균 reward가 대략 0이 되도록
    이동하고, 단위 분산으로 스케일링함. reward model 학습 후 한 번 사전계산됨
    (Phase 2의 calibration 단계 참조).

    참고: 프로덕션 RLHF 시스템은 reward model 출력의 이동 통계를 유지하고
    reward를 온라인으로 정규화함. 정적 calibration이 동일한 메커니즘을
    더 적은 런타임 오버헤드로 구현함.
    """
    features = sequence_to_features(token_ids, vocab_size)
    return reward_forward(features, reward_params)


# === VALUE FUNCTION (PLAIN FLOATS -- NO AUTOGRAD) ===
# Value function은 주어진 시퀀스의 기대 reward를 예측함. advantage 계산에서
# baseline 역할을 함: advantage = reward - value_baseline.
# 이 baseline 없이는 policy gradient의 분산이 높음 (모든 reward 신호가
# 동일하게 정보적인 것으로 취급됨), PPO 최적화가 불안정해짐.
# 참고: 프로덕션 RLHF는 더 깊은 value network와 함께 GAE(Generalized Advantage
# Estimation)를 사용함. 단일 linear layer가 핵심 분산 감소 메커니즘을 구현함.

def init_value_function(d_in: int) -> dict[str, list[float]]:
    """value function 초기화: feature를 scalar로 매핑하는 linear layer."""
    params: dict[str, list[float]] = {}
    params['w'] = [random.gauss(0, 0.01) for _ in range(d_in)]
    params['b'] = [0.0]
    return params


def value_forward(features: list[float], params: dict[str, list[float]]) -> float:
    """value function forward pass: 단순한 dot product + bias."""
    return sum(params['w'][j] * features[j] for j in range(len(features))) + params['b'][0]


def value_update(
    features: list[float],
    target: float,
    params: dict[str, list[float]],
    lr: float,
) -> float:
    """MSE loss로 value function 업데이트: (predicted - target)^2.

    수동 SGD: d_loss/d_w[j] = 2 * (pred - target) * features[j]
    """
    pred = value_forward(features, params)
    error = pred - target
    mse = error ** 2

    # 안정성을 위한 gradient clipping이 포함된 SGD 업데이트
    grad_scale = min(1.0, 1.0 / (abs(error) + 1e-8))
    for j in range(len(features)):
        params['w'][j] -= lr * 2.0 * error * features[j] * grad_scale
    params['b'][0] -= lr * 2.0 * error * grad_scale

    return mse


# === GENERATION AND LOG-PROBABILITY UTILITIES ===

def generate_completion(
    params: dict[str, list[list[Value]]],
    bos: int,
    vocab_size: int,
    max_len: int,
    temperature: float = 0.8,
) -> list[int]:
    """temperature sampling으로 policy model에서 토큰 시퀀스를 생성함.

    생성된 토큰 ID를 반환함(BOS 접두사 제외). 이 함수는 autograd graph를 만들지
    않음 -- sampling에 .data를 사용하며, 생성된 토큰만 필요하고 생성 과정을
    통한 gradient는 필요 없으므로 올바름.
    """
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    token_id = bos
    generated: list[int] = []

    for pos in range(max_len):
        logits = policy_forward(token_id, pos, keys, vals, params)
        # Temperature sampling: 낮은 T = 더 greedy, 높은 T = 더 많은 탐색
        scaled = [logit / temperature for logit in logits]
        probs = softmax(scaled)
        token_id = random.choices(
            range(vocab_size), weights=[p.data for p in probs]
        )[0]
        if token_id == bos:
            break
        generated.append(token_id)

    return generated


def compute_log_probs_detached(
    token_ids: list[int],
    bos: int,
    params: dict[str, list[list[Value]]],
) -> float:
    """policy 하에서 시퀀스의 총 log-probability를 계산함 (autograd graph 없이).

    각 PPO 업데이트 전에 "old" log-prob을 저장하는 데 사용됨. 이것이 importance
    sampling ratio의 분모가 됨: ratio = pi_new(a|s) / pi_old(a|s).
    .data를 사용하여 autograd graph 구축을 피함. scalar 값만 필요하기 때문임.
    """
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    full_seq = [bos] + token_ids
    total_logp = 0.0

    for pos in range(len(token_ids)):
        logits = policy_forward(full_seq[pos], pos, keys, vals, params)
        # 안정적인 log-softmax: log(softmax(x_i)) = x_i - max(x) - log(sum(exp(x_j - max(x))))
        logit_data = [l.data for l in logits]
        max_l = max(logit_data)
        exp_sum = sum(math.exp(l - max_l) for l in logit_data)
        log_prob = logit_data[full_seq[pos + 1]] - max_l - math.log(exp_sum)
        total_logp += log_prob

    return total_logp


def compute_log_probs_autograd(
    token_ids: list[int],
    bos: int,
    params: dict[str, list[list[Value]]],
) -> Value:
    """autograd graph와 함께 시퀀스의 총 log-probability를 계산함.

    PPO 업데이트 내부에서 사용되는 비용이 높은 버전임. PPO가 surrogate objective의
    policy 파라미터에 대한 gradient를 필요로 하므로 autograd graph가 구축되어야 함.
    ratio exp(log_pi_new - log_pi_old)가 이 계산을 통해 흐름.
    """
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    full_seq = [bos] + token_ids
    total_logp: Value = Value(0.0)

    for pos in range(len(token_ids)):
        logits = policy_forward(full_seq[pos], pos, keys, vals, params)
        probs = softmax(logits)
        total_logp = total_logp + safe_log(probs[full_seq[pos + 1]])

    return total_logp


# === TRAINING ===

if __name__ == "__main__":
    start_time = time.time()

    # -- 어휘와 데이터 준비 --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # === Phase 1: Pretraining Policy Model ===
    print("\n=== Phase 1: Pretraining Policy Model ===")
    policy_params = init_policy_params(VOCAB_SIZE)
    policy_param_list = flatten_params(policy_params)
    print(f"Policy parameters: {len(policy_param_list):,} (Value class autograd)")

    # Policy pretraining용 Adam optimizer 상태
    m_pre = [0.0] * len(policy_param_list)
    v_pre = [0.0] * len(policy_param_list)

    for step in range(PRETRAIN_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses: list[Value] = []
        for pos in range(seq_len):
            logits = policy_forward(tokens[pos], pos, keys, vals, policy_params)
            probs = softmax(logits)
            losses.append(-safe_log(probs[tokens[pos + 1]]))

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        # 선형 LR 감쇠가 적용된 Adam
        lr_t = PRETRAIN_LR * (1 - step / PRETRAIN_STEPS)
        for i, p in enumerate(policy_param_list):
            m_pre[i] = BETA1 * m_pre[i] + (1 - BETA1) * p.grad
            v_pre[i] = BETA2 * v_pre[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_pre[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_pre[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{PRETRAIN_STEPS} | loss: {loss.data:.4f}")

    print(f"Pretraining complete. Final loss: {loss.data:.4f}")

    # KL penalty를 위한 reference policy 파라미터 저장. Reference model은 pretraining
    # 종료 시점의 policy -- frozen snapshot임. 프로덕션 RLHF에서는 별도의 모델 복사본이
    # 됨. 여기서는 파라미터 값을 일반 float로 저장하고, reference log-prob 계산 시
    # 일시적으로 교체함.
    ref_param_data: dict[str, list[list[float]]] = {}
    for key, matrix in policy_params.items():
        ref_param_data[key] = [[v.data for v in row] for row in matrix]

    def compute_ref_log_probs(token_ids: list[int]) -> float:
        """frozen reference policy 하에서 log-prob을 계산함.

        policy 파라미터를 일시적으로 reference 값으로 교체하고, log-prob을 계산한 뒤
        (detached -- autograd 없이), 현재 파라미터를 복원함. 두 번째 전체 모델을
        저장하지 않아도 됨. 프로덕션 RLHF는 두 모델 모두 메모리에 유지하지만,
        scalar autograd 객체가 비용이 크므로 메모리 대신 연산을 교환함.
        """
        # 현재 파라미터 저장, reference 파라미터 로드
        current_data: dict[str, list[list[float]]] = {}
        for key, matrix in policy_params.items():
            current_data[key] = [[v.data for v in row] for row in matrix]
            for r, row in enumerate(matrix):
                for c, v in enumerate(row):
                    v.data = ref_param_data[key][r][c]

        logp = compute_log_probs_detached(token_ids, BOS, policy_params)

        # 현재 파라미터 복원
        for key, matrix in policy_params.items():
            for r, row in enumerate(matrix):
                for c, v in enumerate(row):
                    v.data = current_data[key][r][c]

        return logp

    # === Phase 2: Training Reward Model ===
    print("\n=== Phase 2: Training Reward Model ===")

    d_features = VOCAB_SIZE + 1  # 문자 feature + 길이 feature
    reward_params = init_reward_model(d_features)
    reward_param_count = REWARD_HIDDEN * d_features + REWARD_HIDDEN + REWARD_HIDDEN + 1
    print(f"Reward model parameters: {reward_param_count} (plain floats)")

    # names.txt로부터 synthetic preference 쌍을 생성함.
    # "Chosen" = 4-7자 이름 (잘 형성되고 발음 가능한 이름).
    # "Rejected" = 1-3자 이름 (너무 짧음) 또는 8자 이상 이름 (너무 김).
    # 이 synthetic 신호는 적당한 길이의 이름을 선호하는 인간 주석자를 모방함.
    # 3/8 경계(2/10 대신)는 reward model에 선호 범위에 대한 명확한 신호를 줌.
    # 실제 RLHF는 인간 비교를 수집하지만, 동일한 알고리즘 파이프라인을
    # 시연하기 위해 이 휴리스틱을 사용함.
    chosen_names: list[str] = []
    rejected_names: list[str] = []
    for name in docs:
        if 4 <= len(name) <= 7:
            chosen_names.append(name)
        elif len(name) <= 3 or len(name) >= 8:
            rejected_names.append(name)

    random.shuffle(chosen_names)
    random.shuffle(rejected_names)

    # Chosen과 rejected 리스트를 zip하여 preference 쌍 생성
    n_pairs = min(200, len(chosen_names), len(rejected_names))
    preference_pairs: list[tuple[str, str]] = [
        (chosen_names[i], rejected_names[i]) for i in range(n_pairs)
    ]
    print(f"Created {n_pairs} preference pairs")

    # 평가용 20% 홀드아웃
    split = int(0.8 * n_pairs)
    train_pairs = preference_pairs[:split]
    eval_pairs = preference_pairs[split:]

    for step in range(REWARD_STEPS):
        pair = train_pairs[step % len(train_pairs)]
        chosen_tokens = [unique_chars.index(ch) for ch in pair[0]]
        rejected_tokens = [unique_chars.index(ch) for ch in pair[1]]

        feat_c = sequence_to_features(chosen_tokens, VOCAB_SIZE)
        feat_r = sequence_to_features(rejected_tokens, VOCAB_SIZE)

        rloss = reward_backward(feat_c, feat_r, reward_params, REWARD_LR)

        if (step + 1) % 100 == 0 or step == 0:
            correct = 0
            for ep in eval_pairs:
                c_tok = [unique_chars.index(ch) for ch in ep[0]]
                r_tok = [unique_chars.index(ch) for ch in ep[1]]
                sc = reward_forward(sequence_to_features(c_tok, VOCAB_SIZE), reward_params)
                sr = reward_forward(sequence_to_features(r_tok, VOCAB_SIZE), reward_params)
                if sc > sr:
                    correct += 1
            acc = 100.0 * correct / len(eval_pairs)
            print(f"  step {step + 1:>4}/{REWARD_STEPS} | ranking_loss: {rloss:.4f} | accuracy: {acc:.1f}%")

    # 최종 reward model 정확도
    correct = 0
    for ep in eval_pairs:
        c_tok = [unique_chars.index(ch) for ch in ep[0]]
        r_tok = [unique_chars.index(ch) for ch in ep[1]]
        sc = reward_forward(sequence_to_features(c_tok, VOCAB_SIZE), reward_params)
        sr = reward_forward(sequence_to_features(r_tok, VOCAB_SIZE), reward_params)
        if sc > sr:
            correct += 1
    final_acc = 100.0 * correct / len(eval_pairs)
    print(f"Reward model accuracy: {final_acc:.1f}%")

    # Reward model calibration: 학습 분포 시퀀스 샘플에서 평균과 표준편차를
    # 계산하여 PPO 중 reward를 정규화함. 정규화 없이는 절대 reward 스케일이
    # 임의적이어서 advantage 신호의 해석과 KL penalty 계수 튜닝이 어려워짐.
    cal_rewards: list[float] = []
    for name in docs[:200]:
        tok = [unique_chars.index(ch) for ch in name]
        cal_rewards.append(score_completion(tok, VOCAB_SIZE, reward_params))
    reward_mean = sum(cal_rewards) / len(cal_rewards)
    reward_var = sum((r - reward_mean) ** 2 for r in cal_rewards) / len(cal_rewards)
    reward_std = max(math.sqrt(reward_var), 1e-4)

    def normalized_reward(token_ids: list[int]) -> float:
        """학습된 reward model과 length shaping을 결합한 reward를 반환함.

        정규화된 reward model 점수는 pairwise 비교에서 학습한 문자 수준의
        preference를 포착함. length shaping 항은 MLP reward model이 학습 데이터에
        없던 분포 밖 길이(빈 시퀀스, 1자 시퀀스)에 대해 blind spot을 가질 수
        있으므로 선호되는 이름 길이(4-7자)에 대한 명시적 신호를 제공함.

        참고: 프로덕션 RLHF 시스템도 reward shaping(포맷 페널티, 안전성 분류기,
        길이 보너스)을 사용함. length shaping이 동일한 역할을 함.
        """
        n = len(token_ids)
        if n < MIN_GEN_LEN:
            return -3.0

        raw = score_completion(token_ids, VOCAB_SIZE, reward_params)
        norm = (raw - reward_mean) / reward_std

        # Length shaping: 선호 범위에 보너스, 범위 밖에 약한 페널티
        if 4 <= n <= 7:
            length_bonus = 1.0
        elif n == 3:
            length_bonus = 0.0
        else:
            length_bonus = -0.5

        return norm + length_bonus

    # === Phase 3: PPO Training ===
    print("\n=== Phase 3: PPO Training ===")

    value_params = init_value_function(d_features)
    value_param_count = d_features + 1
    print(f"Value function parameters: {value_param_count} (plain floats)")
    print(f"PPO clip epsilon: {PPO_CLIP_EPS} | KL coefficient: {KL_COEFF} (squared penalty)")

    # PPO fine-tuning을 위한 새 Adam 상태 (pretraining momentum을 이어가지 않음,
    # objective가 language modeling에서 reward 최대화로 변경되었으므로)
    m_ppo = [0.0] * len(policy_param_list)
    v_ppo = [0.0] * len(policy_param_list)

    # "PPO 이전" 비교를 위한 pretrained model 상태 저장
    pretrained_param_data: dict[str, list[list[float]]] = {}
    for key, matrix in policy_params.items():
        pretrained_param_data[key] = [[v.data for v in row] for row in matrix]

    # 최종 요약을 위해 PPO 스텝 전체에서 메트릭 추적
    all_rewards: list[float] = []
    all_kl: list[float] = []

    for step in range(PPO_STEPS):
        # --- Step 1: 현재 policy에서 completion 배치 생성 ---
        batch_tokens: list[list[int]] = []
        batch_rewards: list[float] = []
        batch_old_logps: list[float] = []
        batch_ref_logps: list[float] = []
        batch_features: list[list[float]] = []

        for _ in range(BATCH_SIZE):
            gen_tokens = generate_completion(
                policy_params, BOS, VOCAB_SIZE, MAX_GEN_LEN, temperature=0.8
            )
            # 비어있지 않은 completion 보장 (퇴화된 빈 시퀀스는 gradient 신호를 주지 않음)
            if not gen_tokens:
                gen_tokens = [random.randint(0, VOCAB_SIZE - 2)]

            batch_tokens.append(gen_tokens)

            # --- Step 2: 정규화된 reward model로 점수화 ---
            reward = normalized_reward(gen_tokens)
            batch_rewards.append(reward)
            features = sequence_to_features(gen_tokens, VOCAB_SIZE)
            batch_features.append(features)

            # --- Step 5: old log-prob 저장 (파라미터 업데이트 전) ---
            old_logp = compute_log_probs_detached(gen_tokens, BOS, policy_params)
            batch_old_logps.append(old_logp)

            # KL penalty를 위한 reference log-prob
            ref_logp = compute_ref_log_probs(gen_tokens)
            batch_ref_logps.append(ref_logp)

        # --- Step 3: value baseline과 advantage 계산 ---
        batch_advantages: list[float] = []
        for i in range(BATCH_SIZE):
            val = value_forward(batch_features[i], value_params)
            # Advantage = reward - baseline. reward 신호를 중심화하여 policy gradient의
            # 분산을 낮춤: 평균보다 좋은 action은 양의 advantage(강화됨),
            # 평균보다 나쁜 action은 음의 advantage(억제됨).
            # Baseline 없이는 reward > 0인 모든 action이 동일하게 강화됨.
            batch_advantages.append(batch_rewards[i] - val)

        # --- Step 6: PPO 업데이트 (autograd를 통해) ---
        # PPO clipped surrogate objective가 치명적으로 큰 업데이트를 방지함.
        # 수식: L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        # 여기서 ratio = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)
        # 이고 A는 advantage임.
        #
        # 직관: vanilla policy gradient는 ratio * A를 사용하는데, ratio가 클 때
        # (policy가 많이 변한 경우) 거대한 업데이트가 발생할 수 있음. PPO는 ratio를
        # [1-eps, 1+eps]로 clamp하여 스텝 크기를 제한함. 이것이 "proximal" 제약:
        # old policy에 가깝게 유지함.
        total_ppo_loss = Value(0.0)
        total_kl = 0.0

        for i in range(BATCH_SIZE):
            # 현재 log-prob을 autograd와 함께 계산 (비용이 크지만 필수)
            current_logp = compute_log_probs_autograd(batch_tokens[i], BOS, policy_params)

            # Importance sampling ratio: 이 completion에 대해 policy가 얼마나 변했는지
            # ratio = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)
            log_ratio = current_logp - batch_old_logps[i]
            ratio = log_ratio.exp()

            # Clipped surrogate objective
            adv = batch_advantages[i]
            surr1 = ratio * adv                                                    # unclipped
            surr2 = ratio.clip(1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv    # clipped

            # 최솟값을 취함: 보수적임. advantage > 0(좋은 action)일 때 ratio가
            # 1+eps를 초과하지 못하게 함(과도한 강화 방지). advantage < 0(나쁜 action)
            # 일 때 ratio가 1-eps 아래로 내려가지 못하게 함(과도한 페널티 방지).
            # 어느 경우든 업데이트가 제한됨.
            if surr1.data < surr2.data:
                ppo_obj = surr1
            else:
                ppo_obj = surr2

            # KL penalty: policy가 reference에서 너무 멀리 벗어나는 것을 억제함.
            # 이것 없이는 policy가 (불완전한) reward model을 최대화하는 퇴화된
            # 분포로 붕괴됨 -- "reward hacking". KL 항이 reference model의
            # language modeling 품질을 보존함.
            #
            # Squared log-ratio penalty를 사용함: 0.5 * (log_pi - log_pi_ref)^2
            # raw log-ratio(개별 샘플에서 음수가 될 수 있음)와 달리,
            # squared 형태는 항상 >= 0이고 양방향 발산에 페널티를 줌.
            # Schulman (2020)의 "KL penalty" 변형과 동일함.
            kl_per_sample = current_logp.data - batch_ref_logps[i]
            total_kl += abs(kl_per_sample)

            # autograd를 통한 squared KL penalty로 gradient가 policy로 흐름
            log_diff = current_logp - batch_ref_logps[i]
            kl_penalty = KL_COEFF * log_diff * log_diff * 0.5

            # 총 loss: PPO objective를 부정(loss를 최소화하는데 PPO는 objective를 최대화함)
            # + KL penalty 항(항상 양수, policy를 reference 방향으로 밀어줌)
            sample_loss = -ppo_obj + kl_penalty
            total_ppo_loss = total_ppo_loss + sample_loss

        # 배치에 대해 평균
        ppo_loss = total_ppo_loss * (1.0 / BATCH_SIZE)
        avg_kl = total_kl / BATCH_SIZE
        avg_reward = sum(batch_rewards) / BATCH_SIZE

        # Backward pass와 policy 파라미터에 대한 Adam 업데이트
        ppo_loss.backward()

        lr_t = PPO_LR * (1 - step / PPO_STEPS)
        for i, p in enumerate(policy_param_list):
            # Gradient clipping: 다중 스텝 computation graph(PPO 스텝당 배치 전체에서
            # policy forward pass가 여러 번 실행됨)에서의 exploding gradient 방지
            grad = max(-1.0, min(1.0, p.grad))
            m_ppo[i] = BETA1 * m_ppo[i] + (1 - BETA1) * grad
            v_ppo[i] = BETA2 * v_ppo[i] + (1 - BETA2) * grad ** 2
            m_hat = m_ppo[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_ppo[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        # --- Step 7: MSE loss로 value function 업데이트 ---
        for i in range(BATCH_SIZE):
            value_update(batch_features[i], batch_rewards[i], value_params, VALUE_LR)

        all_rewards.append(avg_reward)
        all_kl.append(avg_kl)

        if (step + 1) % 20 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{PPO_STEPS} | reward: {avg_reward:.2f} | "
                  f"kl_div: {avg_kl:.2f} | ppo_loss: {ppo_loss.data:.4f}")

    # === Results ===
    print("\n=== Results ===")

    # pretrained model(PPO 이전)에서 생성하기 위해 일시적으로 가중치 복원
    current_data_backup: dict[str, list[list[float]]] = {}
    for key, matrix in policy_params.items():
        current_data_backup[key] = [[v.data for v in row] for row in matrix]
        for r, row in enumerate(matrix):
            for c, v in enumerate(row):
                v.data = pretrained_param_data[key][r][c]

    print("Generating from PRETRAINED model (before PPO):")
    pre_rewards: list[float] = []
    pre_lengths: list[int] = []
    for i in range(10):
        gen = generate_completion(policy_params, BOS, VOCAB_SIZE, MAX_GEN_LEN, temperature=0.5)
        name = ''.join(unique_chars[t] for t in gen if t < len(unique_chars))
        shaped_r = normalized_reward(gen)
        pre_rewards.append(shaped_r)
        pre_lengths.append(len(gen))
        print(f"  {i + 1:>2}. {name:10s} (reward: {shaped_r:+.2f}, len: {len(gen)})")

    # PPO로 학습된 가중치 복원
    for key, matrix in policy_params.items():
        for r, row in enumerate(matrix):
            for c, v in enumerate(row):
                v.data = current_data_backup[key][r][c]

    print("\nGenerating from PPO-ALIGNED model:")
    post_rewards: list[float] = []
    post_lengths: list[int] = []
    for i in range(10):
        gen = generate_completion(policy_params, BOS, VOCAB_SIZE, MAX_GEN_LEN, temperature=0.5)
        name = ''.join(unique_chars[t] for t in gen if t < len(unique_chars))
        shaped_r = normalized_reward(gen)
        post_rewards.append(shaped_r)
        post_lengths.append(len(gen))
        print(f"  {i + 1:>2}. {name:10s} (reward: {shaped_r:+.2f}, len: {len(gen)})")

    avg_pre = sum(pre_rewards) / len(pre_rewards)
    avg_post = sum(post_rewards) / len(post_rewards)
    avg_pre_len = sum(pre_lengths) / len(pre_lengths)
    avg_post_len = sum(post_lengths) / len(post_lengths)
    avg_kl_final = sum(all_kl[-20:]) / len(all_kl[-20:]) if all_kl else 0.0

    print(f"\nAverage reward -- Before PPO: {avg_pre:+.2f} | After PPO: {avg_post:+.2f}")
    print(f"Average length -- Before PPO: {avg_pre_len:.1f} | After PPO: {avg_post_len:.1f}")
    print(f"Average KL divergence from reference: {avg_kl_final:.2f}")

    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.1f}s")
