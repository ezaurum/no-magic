"""
Direct Preference Optimization (DPO): 단일 contrastive loss로 language model을 human preference에
정렬하는 방법 — reward model 없음, reinforcement learning 없음, preference 쌍에 대한 supervised learning만 사용함.
"""
# Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model is
# Secretly a Reward Model" (2023). https://arxiv.org/abs/2305.18290
# microgpt 패턴(Radford et al., 2019)을 재사용하며 교육적 목적으로 단순화함:
# RMSNorm, ReLU, bias 없음.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# 모델 아키텍처 — microgpt와 동일해서 직접 비교 가능함
N_EMBD = 16         # embedding 차원 (d_model)
N_HEAD = 4          # attention head 수
N_LAYER = 1         # transformer 블록 수
BLOCK_SIZE = 16     # context window 길이
HEAD_DIM = N_EMBD // N_HEAD  # head당 4차원

# 학습 — base model pretraining
BASE_LR = 0.01
BASE_STEPS = 700

# 학습 — DPO alignment
DPO_LR = 0.003
DPO_STEPS = 60
DPO_BETA = 0.1
# beta는 alignment 강도를 제어함. 낮은 beta(0.01)는 policy를 reference에서 거의 안 움직이고;
# 높은 beta(1.0)는 분포를 preferred completion 쪽으로 공격적으로 변형하지만 mode collapse 위험이 있음.
# 0.1은 논문의 표준 시작점임.
# 직관적으로: beta는 implicit reward model의 inverse temperature — beta가 높을수록
# preference가 날카로워짐.

# 공유 optimizer 상수
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8

# 데이터
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 추론
TEMPERATURE = 0.5
NUM_SAMPLES = 10

# 참고: 총 ~4,200개 parameter. 실제 DPO(Llama-2-Chat, Zephyr)는 수십억 parameter의 모델을
# 수천 개의 human-labeled preference 쌍으로 정렬함. 알고리즘은 동일하고 스케일만 다름.


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
    """reverse-mode 자동 미분을 지원하는 스칼라 값.

    모든 forward 연산은 local derivative(dout/dinput)를 기록함. backward()는
    computation graph를 역 위상 정렬 순서로 재생하며, chain rule을 통해 gradient를 누적함:
    dLoss/dx = 모든 경로에 대해 (경로를 따른 local gradient의 곱)의 합.
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
# 이 Value 클래스는 canonical interface를 따름 (docs/autograd-interface.md 참조).
# base set 이상의 추가 기능 없음. safe_log()는 log-probability 계산의
# 수치 안정성을 위해 사용됨.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """weight 행렬을 ~ N(0, std)로 초기화함. 표준편차 0.08은 이 tiny model에 대해
    경험적으로 튜닝된 값임; 큰 모델은 Xavier/Glorot 스케일링(std = 1/sqrt(d_in))을 사용함."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_parameters(vocab_size: int) -> dict[str, list[list[Value]]]:
    """모든 모델 parameter를 초기화함: embedding, attention, MLP, LM head."""
    params: dict[str, list[list[Value]]] = {}

    params['wte'] = make_matrix(vocab_size, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    for layer_idx in range(N_LAYER):
        params[f'layer{layer_idx}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)

        # MLP: 4배 확장 후 축소 (feedforward 용량을 위한 GPT 관례)
        params[f'layer{layer_idx}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)

    params['lm_head'] = make_matrix(vocab_size, N_EMBD)

    return params


def flatten_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """parameter dict에서 모든 Value 객체를 flat list로 모음."""
    return [p for matrix in params.values() for row in matrix for p in row]


def snapshot_weights(params: dict[str, list[list[Value]]]) -> dict[str, list[list[float]]]:
    """reference model용으로 모든 parameter 값을 plain float로 deep copy함.

    reference model은 pretrained policy의 고정된 스냅샷임. DPO는 이걸로
    log-probability ratio를 계산함: policy가 시작점에서 얼마나 벗어났는가?
    plain float(Value 객체가 아닌)로 저장하는 게 중요한 이유 두 가지:
    1. autograd 오버헤드 없음 — reference forward pass가 그래프 생성 없이 ~10배 빠름
    2. gradient 오염 없음 — reference model은 절대로 gradient 업데이트를 받으면 안 됨
    """
    return {
        key: [[v.data for v in row] for row in matrix]
        for key, matrix in params.items()
    }


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """행렬-벡터 곱: y = W @ x. W의 shape이 [n_out, n_in]이고 x의 shape이
    [n_in]일 때, 출력 y의 shape은 [n_out]이며 y[i] = sum_j W[i,j] * x[j]."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def linear_float(x: list[float], w: list[list[float]]) -> list[float]:
    """plain float를 사용한 행렬-벡터 곱. linear()와 동일하지만 reference model
    forward pass용으로 raw float에서 동작함 — autograd 그래프 생성 없음."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """안정적인 softmax: overflow 방지를 위해 exp 전에 max를 뺌.
    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))"""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def softmax_float(logits: list[float]) -> list[float]:
    """reference model용 plain float에 대한 안정적인 softmax."""
    max_val = max(logits)
    exp_vals = [math.exp(v - max_val) for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMS 정규화: x / sqrt(mean(x^2) + eps).
    LayerNorm보다 단순함 (평균 중심화 없음, 학습 가능한 affine 없음). LLaMA, Gemma에서 사용됨."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def rmsnorm_float(x: list[float]) -> list[float]:
    """reference model용 plain float에 대한 RMS 정규화."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """수치 안정성을 위한 클리핑된 log. log(0) = -inf를 방지해서 gradient 전파가
    깨지지 않게 함. prob을 child로 하여 노드를 수동으로 생성하므로
    gradient가 computation graph를 통해 역전파됨 (clamping으로 끊기지 않음)."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === GPT FORWARD PASS (POLICY MODEL — AUTOGRAD) ===

def gpt_forward(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
) -> list[Value]:
    """policy model을 통한 단일 토큰 forward pass (autograd 포함).

    vocabulary에 대한 logit을 반환함. key/value는 explicit mask 없이
    causal attention을 위한 KV cache를 누적함.
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
        v_proj = linear(x, params[f'layer{layer_idx}.attn_wv'])

        keys[layer_idx].append(k)
        values[layer_idx].append(v_proj)

        x_attn: list[Value] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_head = q[hs : hs + HEAD_DIM]
            k_head = [k_t[hs : hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[hs : hs + HEAD_DIM] for v_t in values[layer_idx]]

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

        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === GPT FORWARD PASS (REFERENCE MODEL — PLAIN FLOATS) ===

def gpt_forward_float(
    token_id: int,
    pos_id: int,
    keys: list[list[list[float]]],
    values: list[list[list[float]]],
    params: dict[str, list[list[float]]],
) -> list[float]:
    """고정된 reference model용 plain float를 사용한 단일 토큰 forward pass.

    gpt_forward()와 구조적으로 동일하지만 전부 float로 동작함.
    computation graph를 생성하지 않아서 ~10배 빠름. reference model은
    절대 업데이트되지 않으므로 gradient가 불필요함.
    """
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm_float(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm_float(x)

        q = linear_float(x, params[f'layer{layer_idx}.attn_wq'])
        k = linear_float(x, params[f'layer{layer_idx}.attn_wk'])
        v_proj = linear_float(x, params[f'layer{layer_idx}.attn_wv'])

        keys[layer_idx].append(k)
        values[layer_idx].append(v_proj)

        x_attn: list[float] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_head = q[hs : hs + HEAD_DIM]
            k_head = [k_t[hs : hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[hs : hs + HEAD_DIM] for v_t in values[layer_idx]]

            attn_logits = [
                sum(q_head[j] * k_head[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]
            attn_weights = softmax_float(attn_logits)

            head_output = [
                sum(attn_weights[t] * v_head[t][j] for t in range(len(v_head)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_output)

        x = linear_float(x_attn, params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        x = rmsnorm_float(x)
        x = linear_float(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [max(0.0, xi) for xi in x]  # plain float에 대한 ReLU
        x = linear_float(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear_float(x, params['lm_head'])


# === SEQUENCE LOG-PROBABILITY ===

def sequence_log_prob_policy(
    tokens: list[int],
    params: dict[str, list[list[Value]]],
) -> Value:
    """autograd를 사용해서 policy model 하에서 log P(sequence)를 계산함.

    Math: log P(x_0, x_1, ..., x_T) = sum_{t=0}^{T-1} log P(x_{t+1} | x_0..x_t)
    각 항은 이전 토큰들이 주어졌을 때 다음 토큰의 log-probability임.
    합이 autograd 그래프를 통해 흐르므로 DPO gradient가 모든 parameter에 도달함.
    """
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    total_log_prob = Value(0.0)

    for pos in range(len(tokens) - 1):
        logits = gpt_forward(tokens[pos], pos, keys, vals, params)
        probs = softmax(logits)
        total_log_prob = total_log_prob + safe_log(probs[tokens[pos + 1]])

    return total_log_prob


def sequence_log_prob_reference(
    tokens: list[int],
    ref_params: dict[str, list[list[float]]],
) -> float:
    """plain float를 사용해서 고정된 reference model 하에서 log P(sequence)를 계산함.

    수학은 sequence_log_prob_policy와 동일하지만 autograd 오버헤드가 없음. reference model은
    절대 업데이트되지 않으므로 plain float를 반환함 — 그 log-prob은 DPO loss에서 상수임.
    """
    keys: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
    vals: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
    total_log_prob = 0.0

    for pos in range(len(tokens) - 1):
        logits = gpt_forward_float(tokens[pos], pos, keys, vals, ref_params)
        probs = softmax_float(logits)
        prob_next = max(probs[tokens[pos + 1]], 1e-10)
        total_log_prob += math.log(prob_next)

    return total_log_prob


# === DPO LOSS ===

def dpo_loss(
    chosen_tokens: list[int],
    rejected_tokens: list[int],
    params: dict[str, list[list[Value]]],
    ref_params: dict[str, list[list[float]]],
    beta: float,
) -> tuple[Value, float, float]:
    """단일 preference 쌍에 대한 DPO loss를 계산함.

    Math: L_DPO = -log(sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))

    여기서:
        y_w = chosen (선호된) completion
        y_l = rejected completion
        pi = policy model (학습 가능)
        pi_ref = reference model (고정된 pretrained 스냅샷)
        sigma = sigmoid 함수
        beta = alignment 강도 (implicit reward model의 inverse temperature)

    log-ratio log(pi/pi_ref)는 주어진 시퀀스에 대해 policy가 reference에서 얼마나
    벗어났는지를 측정함. DPO는 이 ratio를 chosen 시퀀스에 대해서는 올리고
    rejected 시퀀스에 대해서는 내림. Rafailov et al.의 핵심 통찰: 이 contrastive
    objective는 implicit reward model r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))을 사용한
    RL과 동치이지만, 그 reward model을 명시적으로 학습할 필요가 없음.

    반환값: (loss, chosen_reward, rejected_reward) — reward는 alignment 진행 상황
    모니터링에 사용되는 implicit reward 값임.
    """
    # Policy log-prob (autograd 그래프를 통해)
    log_pi_chosen = sequence_log_prob_policy(chosen_tokens, params)
    log_pi_rejected = sequence_log_prob_policy(rejected_tokens, params)

    # Reference log-prob (plain float, gradient 불필요)
    log_ref_chosen = sequence_log_prob_reference(chosen_tokens, ref_params)
    log_ref_rejected = sequence_log_prob_reference(rejected_tokens, ref_params)

    # Log-ratio: policy가 reference에서 얼마나 벗어났는가?
    # Math: log(pi(y|x) / pi_ref(y|x)) = log pi(y|x) - log pi_ref(y|x)
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    # Implicit reward 차이 (sigmoid의 인자)
    # Math: delta = beta * (log_ratio_chosen - log_ratio_rejected)
    # 양의 delta는 policy가 reference보다 chosen을 rejected 대비 더 선호한다는 뜻임.
    delta = beta * (log_ratio_chosen - log_ratio_rejected)

    # 수치적으로 안정적인 sigmoid: -log(sigma(x)) = log(1 + exp(-x))
    # 큰 양의 x: log(1 + exp(-x)) ~ exp(-x) ~ 0 (loss 거의 0, 올바른 preference)
    # 큰 음의 x: log(1 + exp(-x)) ~ -x (loss 증가, 잘못된 preference)
    # 구현: loss = -log(sigma(delta)) = log(1 + exp(-delta))
    # logsigmoid 항등식을 사용해서 overflow할 수 있는 sigma 직접 계산을 피함.
    neg_delta = -delta
    # 안정적인 log(1 + exp(z)): z >> 0이면 z 자체, z << 0이면 exp(z) 사용
    if neg_delta.data > 20.0:
        # 매우 큰 neg_delta의 경우 exp(-delta)가 지배함: log(1+exp(z)) ~ z
        loss = neg_delta
    else:
        loss = (Value(1.0) + neg_delta.exp()).log()

    # 모니터링용 implicit reward (plain float, loss 그래프에 포함되지 않음)
    # Math: r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))
    # DPO policy가 각 completion에 암묵적으로 부여하는 reward.
    chosen_reward = beta * log_ratio_chosen.data
    rejected_reward = beta * log_ratio_rejected.data

    return loss, chosen_reward, rejected_reward


# === PREFERENCE PAIR CONSTRUCTION ===

def create_preference_pairs(
    docs: list[str],
    unique_chars: list[str],
    bos: int,
    min_prompt_len: int = 2,
    max_prompt_len: int = 3,
) -> list[tuple[list[int], list[int]]]:
    """학습 데이터에서 synthetic preference 쌍을 생성함.

    전략: 긴 이름을 짧은 이름보다 선호함. 각 이름에서 처음 2-3글자를 공유 prompt
    prefix로 사용하고, 같은 prefix를 공유하는 긴 completion(chosen)과
    짧은 completion(rejected)을 쌍으로 만듦.

    왜 길이를 preference 신호로 사용하는가? 단순하고, 검증 가능하며, human annotation이
    필요 없음. DPO 학습 후 policy는 reference model보다 긴 이름을 생성해야 함
    — 알고리즘이 작동함을 증명하는 측정 가능한 행동 변화임.

    실제 DPO는 human-labeled preference 쌍을 사용함 (예: "응답 A가 응답 B보다 더 도움됨").
    loss 함수는 동일하고 preference 신호만 다름.
    """
    # prefix(처음 2-3글자) 기준으로 이름을 그룹화
    from collections import defaultdict
    prefix_groups: dict[str, list[str]] = defaultdict(list)

    for doc in docs:
        if len(doc) < 2:
            continue
        for plen in range(min_prompt_len, min(max_prompt_len + 1, len(doc))):
            prefix = doc[:plen]
            prefix_groups[prefix].append(doc)

    pairs: list[tuple[list[int], list[int]]] = []

    for prefix, names in prefix_groups.items():
        # 긴 이름(chosen)과 짧은 이름(rejected)으로 분리
        long_names = [n for n in names if len(n) >= 5]
        short_names = [n for n in names if len(n) <= 3]

        if not long_names or not short_names:
            continue

        # 쌍 생성: 각 긴 이름을 같은 prefix를 공유하는 짧은 이름과 짝지음
        for long_name in long_names[:2]:  # prefix당 과도한 쌍을 피하기 위해 제한
            short_name = random.choice(short_names)

            # 토큰화: BOS 마커를 포함한 전체 시퀀스
            chosen_tokens = [bos] + [unique_chars.index(ch) for ch in long_name] + [bos]
            rejected_tokens = [bos] + [unique_chars.index(ch) for ch in short_name] + [bos]

            # block size에 맞게 자름
            chosen_tokens = chosen_tokens[:BLOCK_SIZE + 1]
            rejected_tokens = rejected_tokens[:BLOCK_SIZE + 1]

            pairs.append((chosen_tokens, rejected_tokens))

    random.shuffle(pairs)
    return pairs


# === GENERATION ===

def generate_names(
    params: dict[str, list[list[Value]]],
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    num_samples: int = 10,
    temperature: float = 0.5,
) -> list[str]:
    """모델에서 autoregressive 샘플링으로 이름을 생성함."""
    results: list[str] = []
    for _ in range(num_samples):
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        token_id = bos
        generated: list[str] = []
        for pos in range(BLOCK_SIZE):
            logits = gpt_forward(token_id, pos, keys, vals, params)
            scaled = [logit / temperature for logit in logits]
            probs = softmax(scaled)
            token_id = random.choices(
                range(vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == bos:
                break
            generated.append(unique_chars[token_id])
        results.append(''.join(generated))
    return results


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

    # =========================================================================
    # === Phase 1: Pretraining Base Model ===
    # =========================================================================
    # 표준 language model 학습: 이전 문자들이 주어졌을 때 다음 문자를 예측함.
    # 이것이 DPO가 나중에 정렬할 base policy를 생성함.
    print("\n=== Phase 1: Pretraining Base Model ===")
    params = init_parameters(VOCAB_SIZE)
    param_list = flatten_params(params)
    print(f"Parameters: {len(param_list):,}")

    m_state = [0.0] * len(param_list)
    v_state = [0.0] * len(param_list)

    for step in range(BASE_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses: list[Value] = []
        for pos in range(seq_len):
            logits = gpt_forward(tokens[pos], pos, keys, vals, params)
            probs = softmax(logits)
            losses.append(-safe_log(probs[tokens[pos + 1]]))

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        lr_t = BASE_LR * (1 - step / BASE_STEPS)
        for i, p in enumerate(param_list):
            m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * p.grad
            v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{BASE_STEPS} | loss: {loss.data:.4f}")

    print(f"Pretraining complete. Final loss: {loss.data:.4f}")

    # =========================================================================
    # === Phase 2: Creating Preference Pairs ===
    # =========================================================================
    # preference 쌍을 만들기 전에 pretrained weight를 고정된 reference model로 스냅샷함.
    # reference model은 DPO loss를 고정시킴 — policy가 pretrained 분포에서
    # 너무 멀리 벗어나는 것을 방지함 (KL regularization).
    print("\n=== Phase 2: Creating Preference Pairs ===")

    ref_params = snapshot_weights(params)

    preference_pairs = create_preference_pairs(docs, unique_chars, BOS)

    # 실행 시간을 제한하기 위해 쌍의 수를 제한함. 실제 DPO는 수천~수백만 개의 쌍을 사용함;
    # 각 쌍이 두 번의 전체 시퀀스 forward pass(policy + reference)를 필요로 하므로 150개를 사용함.
    max_pairs = 150
    if len(preference_pairs) > max_pairs:
        preference_pairs = preference_pairs[:max_pairs]

    print(f"Created {len(preference_pairs)} preference pairs (prefer longer completions)")

    # 해석 가능성을 위해 예시 쌍을 보여줌
    if preference_pairs:
        chosen_ex, rejected_ex = preference_pairs[0]
        chosen_str = ''.join(unique_chars[t] for t in chosen_ex[1:-1])  # BOS 제거
        rejected_str = ''.join(unique_chars[t] for t in rejected_ex[1:-1])
        print(f'Example: chosen="{chosen_str}" | rejected="{rejected_str}"')

    # =========================================================================
    # === Phase 3: DPO Training ===
    # =========================================================================
    # 핵심 DPO 루프. 각 preference 쌍(chosen, rejected)에 대해:
    # 1. policy 하에서 log P(chosen)과 log P(rejected)를 계산 (autograd 포함)
    # 2. reference 하에서 log P(chosen)과 log P(rejected)를 계산 (plain float)
    # 3. DPO loss가 policy를 밀어서 log-ratio 격차를 넓힘: reference가 하는 것 대비
    #    chosen을 더 선호하고 rejected를 더 기피하게 만듦.
    #
    # 왜 RLHF(PPO) 대신 DPO인가?
    # 표준 RLHF는 다음이 필요함: (1) reward model 학습, (2) reward model을 신호로 PPO 실행.
    # DPO는 reward model 하의 최적 policy가 preference 데이터와 closed-form 관계를 가짐을
    # 증명해서 두 단계를 하나의 supervised loss로 합침. 결과적으로 코드가 더 단순하고,
    # hyperparameter가 적고, reward model이 필요 없음.
    print("\n=== Phase 3: DPO Training ===")
    print(f"Beta: {DPO_BETA}")

    # DPO 단계용으로 optimizer 상태를 리셋함 (pretraining과 학습 dynamics가 다름)
    m_state = [0.0] * len(param_list)
    v_state = [0.0] * len(param_list)

    for step in range(DPO_STEPS):
        # preference 쌍의 mini-batch를 샘플링
        # batch를 사용하면 gradient 추정의 분산이 줄어듦
        batch_size = 4
        batch_indices = [random.randint(0, len(preference_pairs) - 1) for _ in range(batch_size)]

        total_loss = Value(0.0)
        total_chosen_reward = 0.0
        total_rejected_reward = 0.0

        for idx in batch_indices:
            chosen_tokens, rejected_tokens = preference_pairs[idx]
            loss_i, cr, rr = dpo_loss(
                chosen_tokens, rejected_tokens, params, ref_params, DPO_BETA
            )
            total_loss = total_loss + loss_i
            total_chosen_reward += cr
            total_rejected_reward += rr

        # batch에 대해 평균
        avg_loss = total_loss * (1.0 / batch_size)
        avg_loss.backward()

        lr_t = DPO_LR * (1 - step / DPO_STEPS)
        for i, p in enumerate(param_list):
            m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * p.grad
            v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 10 == 0 or step == 0:
            mean_cr = total_chosen_reward / batch_size
            mean_rr = total_rejected_reward / batch_size
            print(
                f"  step {step + 1:>3}/{DPO_STEPS} | "
                f"dpo_loss: {avg_loss.data:.4f} | "
                f"mean_chosen_reward: {mean_cr:.2f} | "
                f"mean_rejected_reward: {mean_rr:.2f}"
            )

    print("DPO training complete.")

    # =========================================================================
    # === Results ===
    # =========================================================================
    # reference model(pretrained)과 DPO-aligned policy의 생성 결과를 비교함.
    # aligned model은 평균적으로 더 긴 이름을 생성해야 함 — 그것이 preference 신호임.
    print("\n=== Results ===")

    # reference weight를 새 Value 객체에 임시 로드해서 reference model에서 생성함
    # (generation 함수에 Value 객체가 필요함).
    ref_value_params: dict[str, list[list[Value]]] = {
        key: [[Value(v) for v in row] for row in matrix]
        for key, matrix in ref_params.items()
    }

    print("Generating from REFERENCE model:")
    ref_names = generate_names(ref_value_params, unique_chars, BOS, VOCAB_SIZE, NUM_SAMPLES, TEMPERATURE)
    for i, name in enumerate(ref_names):
        print(f"  {i + 1:>2}. {name} (length {len(name)})")

    print("\nGenerating from DPO-ALIGNED model:")
    dpo_names = generate_names(params, unique_chars, BOS, VOCAB_SIZE, NUM_SAMPLES, TEMPERATURE)
    for i, name in enumerate(dpo_names):
        print(f"  {i + 1:>2}. {name} (length {len(name)})")

    ref_avg = sum(len(n) for n in ref_names) / len(ref_names) if ref_names else 0
    dpo_avg = sum(len(n) for n in dpo_names) / len(dpo_names) if dpo_names else 0
    print(f"\nAverage generated length -- Reference: {ref_avg:.1f} | DPO-aligned: {dpo_avg:.1f}")

    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.1f}s")
