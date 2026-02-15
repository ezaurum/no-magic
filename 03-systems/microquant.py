"""
최소한의 품질 손실로 모델을 4배 줄이는 방법 -- INT8과 INT4 가중치 quantization의
수학을 엔드투엔드로 시연함: 학습, 양자화, 역양자화, 비교.
"""
# Reference: Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers
# at Scale" (2022). https://arxiv.org/abs/2208.07339
# Also: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
# Pre-trained Transformers" (2022). https://arxiv.org/abs/2210.17323

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# 모델 아키텍처 (일관성을 위해 microgpt와 동일)
N_EMBD = 16
N_HEAD = 4
N_LAYER = 1
BLOCK_SIZE = 16
HEAD_DIM = N_EMBD // N_HEAD  # 4

# 학습 파라미터
LEARNING_RATE = 0.01
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
NUM_STEPS = 800

# 데이터 파라미터
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 참고: 800 스텝(microgpt의 1000 대비)이면 충분함. 이 스크립트의 초점은
# quantization이지 학습 loss를 밀어붙이는 게 아니기 때문임. 수렴된 모델이 필요하지 최적 모델은 아님.


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
    """역전파 자동 미분이 있는 스칼라 값.

    ._children과 ._local_grads를 통해 계산 이력을 추적해서
    체인 룰로 gradient를 계산할 수 있게 함. 모든 순전파 연산이
    로컬 도함수(dout/dinput)를 저장하고, backward()가 그래프를
    역 위상 정렬 순서로 재생하면서 gradient를 누적함.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent):
        # d(x^n)/dx = n * x^(n-1)
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1)
    def __rtruediv__(self, other): return other * (self ** -1)

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """계산 그래프의 위상 정렬을 통한 역전파 자동 미분."""
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# 이 Value 클래스는 정규 인터페이스를 정확히 따름.
# 전체 스펙은 docs/autograd-interface.md 참조.
# autograd는 학습에만 사용됨. quantization과 추론은 일반 float를 사용함.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """가우시안 초기화된 가중치 행렬. std=0.08은 이 작은 모델에 적합함."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


# === CORE OPERATIONS (AUTOGRAD) ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """y = W @ x. 근본적인 신경망 연산임."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """안정적인 softmax: exp(x - max(x)) / sum(exp(x_j - max(x_j)))."""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps). 학습 가능한 어파인 파라미터 없음."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """gradient 깨짐을 방지하기 위해 클리핑된 log. log(0) = -inf를 막음."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === CORE OPERATIONS (PLAIN FLOAT -- for quantized inference) ===
# quantization 후 가중치는 float로 역양자화됨. 이 함수들은
# autograd 버전과 동일하지만 원시 float로 동작함 — 양자화된
# 모델은 추론 전용이라 gradient 추적이 없음.

def linear_float(x: list[float], w: list[list[float]]) -> list[float]:
    """y = W @ x를 일반 float로 계산함. quantization 후 추론에 사용됨."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax_float(logits: list[float]) -> list[float]:
    """일반 float logits에 대한 안정적인 softmax."""
    max_val = max(logits)
    exp_vals = [math.exp(v - max_val) for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm_float(x: list[float]) -> list[float]:
    """일반 float에 대한 RMSNorm."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# === GPT FORWARD PASS (AUTOGRAD) ===

def gpt_forward(
    token_id: int, pos_id: int,
    keys: list[list[list[Value]]], values: list[list[list[Value]]],
    params: dict,
) -> list[Value]:
    """GPT를 통한 단일 토큰 순전파. 어휘 logits을 반환함."""
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

        x_attn = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_h = q[hs:hs + HEAD_DIM]
            k_h = [k_t[hs:hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_h = [v_t[hs:hs + HEAD_DIM] for v_t in values[layer_idx]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)

        x = linear(x_attn, params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === GPT FORWARD PASS (PLAIN FLOAT -- for quantized inference) ===
# gpt_forward와 구조적으로 동일하지만 역양자화된 float 가중치로 동작함.
# 이 분리로 quantization 평가 경로에서 autograd 오버헤드를 제거함.

def gpt_forward_float(
    token_id: int, pos_id: int,
    keys: list[list[list[float]]], values: list[list[list[float]]],
    float_params: dict[str, list[list[float]]],
) -> list[float]:
    """일반 float 가중치를 사용한 단일 토큰 순전파. gradient 추적 없음."""
    tok_emb = float_params['wte'][token_id]
    pos_emb = float_params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm_float(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm_float(x)
        q = linear_float(x, float_params[f'layer{layer_idx}.attn_wq'])
        k = linear_float(x, float_params[f'layer{layer_idx}.attn_wk'])
        v = linear_float(x, float_params[f'layer{layer_idx}.attn_wv'])
        keys[layer_idx].append(k)
        values[layer_idx].append(v)

        x_attn: list[float] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_h = q[hs:hs + HEAD_DIM]
            k_h = [k_t[hs:hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_h = [v_t[hs:hs + HEAD_DIM] for v_t in values[layer_idx]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_h))
            ]
            attn_weights = softmax_float(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)

        x = linear_float(x_attn, float_params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm_float(x)
        x = linear_float(x, float_params[f'layer{layer_idx}.mlp_fc1'])
        x = [max(0.0, xi) for xi in x]  # 일반 float에 대한 ReLU
        x = linear_float(x, float_params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear_float(x, float_params['lm_head'])


# === QUANTIZATION FUNCTIONS ===
# quantization은 연속적인 float 가중치를 이산 정수 격자에 매핑함.
# 핵심 통찰: 신경망 가중치는 대략 정규 분포이며 크기가 작음.
# 대부분의 값이 0 근처에 모여 있어서, [-127, +127] (INT8)이나
# [-8, +7] (INT4)로 매핑해도 의외로 적은 정보만 손실됨. 네트워크의
# 비선형성과 중복성이 반올림 오차를 흡수함.

def quantize_absmax_int8(weights_float: list[list[float]]) -> tuple[list[list[int]], float]:
    """Absmax quantization: scale = max(|W|) / 127, q = round(W / scale).

    float 범위 [-max|W|, +max|W|]를 정수 범위 [-127, +127]로 매핑함.
    0을 중심으로 대칭 — 가중치 분포가 대략 중심에 있다고 가정함.
    가장 단순한 quantization 방식이며 다른 모든 것의 기준선임.

    수식: q_i = clamp(round(w_i / s), -127, 127)  여기서 s = max(|W|) / 127
    역양자화: w_hat_i = q_i * s
    """
    max_abs = max(abs(w) for row in weights_float for w in row)
    if max_abs == 0:
        return [[0] * len(row) for row in weights_float], 1.0
    scale = max_abs / 127.0
    quantized = [[max(-127, min(127, round(w / scale))) for w in row] for row in weights_float]
    return quantized, scale


def quantize_absmax_int4(weights_float: list[list[float]]) -> tuple[list[list[int]], float]:
    """INT4 quantization: [-8, +7] (4비트 부호 있는 정수 범위)로 매핑함.

    float32 대비 8배 압축. quantization 격자가 INT8보다 16배 거칠어서
    반올림 오차가 상당히 커짐. 신경망이 이걸 견디는 이유는 개별 가중치
    정밀도보다 가중치 행렬의 전체적 통계적 특성이 더 중요하기 때문임.

    참고: 프로덕션 INT4 (GPTQ, AWQ)는 단순 반올림 대신 보정 데이터를 사용해서
    출력 오차를 최소화함. 이렇게 하면 품질 손실이 2-5배 줄지만
    보정 데이터셋이 필요함.
    """
    max_abs = max(abs(w) for row in weights_float for w in row)
    if max_abs == 0:
        return [[0] * len(row) for row in weights_float], 1.0
    scale = max_abs / 7.0
    quantized = [[max(-8, min(7, round(w / scale))) for w in row] for row in weights_float]
    return quantized, scale


def quantize_zeropoint_int8(
    weights_float: list[list[float]],
) -> tuple[list[list[int]], float, int]:
    """Zero-point (비대칭) quantization: [min_W, max_W]를 [0, 255]로 매핑함.

    0을 중심으로 하는 absmax와 달리, zero-point는 전체 8비트 범위가
    실제 가중치 범위를 커버하도록 매핑을 시프트함. 가중치가 0을 중심으로
    대칭이 아닐 때 더 정확함 (ReLU 위주 아키텍처에서 바이어스가 분포를
    이동시킨 경우에 흔함).

    수식: scale = (w_max - w_min) / 255
          zero_point = round(-w_min / scale)
          q_i = clamp(round(w_i / scale) + zero_point, 0, 255)
    역양자화: w_hat_i = (q_i - zero_point) * scale
    """
    all_weights = [w for row in weights_float for w in row]
    w_min = min(all_weights)
    w_max = max(all_weights)
    if w_max == w_min:
        return [[0] * len(row) for row in weights_float], 1.0, 0
    scale = (w_max - w_min) / 255.0
    zero_point = round(-w_min / scale)
    quantized = [
        [max(0, min(255, round(w / scale) + zero_point)) for w in row]
        for row in weights_float
    ]
    return quantized, scale, zero_point


def quantize_per_channel_int8(
    weights_float: list[list[float]],
) -> tuple[list[list[int]], list[float]]:
    """Per-channel quantization: 각 출력 행이 자체 스케일 팩터를 가짐.

    Per-tensor quantization은 전체 행렬에 하나의 스케일을 사용하므로,
    하나의 이상치 가중치가 전체 격자를 거칠게 만듦. Per-channel (per-row)
    quantization은 각 출력 뉴런이 자체 범위를 사용해서, 행 크기가
    불균일한 행렬에서 오차를 크게 줄임.

    참고: LLM.int8() (Dettmers 2022)는 혼합 정밀도 분해로 더 나아감
    -- 이상치 채널은 fp16으로 유지하고 나머지만 INT8로 양자화함.
    """
    quantized = []
    scales = []
    for row in weights_float:
        max_abs = max(abs(w) for w in row)
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        scales.append(scale)
        quantized.append([max(-127, min(127, round(w / scale))) for w in row])
    return quantized, scales


# === DEQUANTIZATION FUNCTIONS ===
# quantization 매핑을 역으로 적용해서 근사 float 가중치를 복구함.
# 원본과 역양자화된 가중치의 차이가 quantization 오차임.

def dequantize_absmax(quantized: list[list[int]], scale: float) -> list[list[float]]:
    """w_hat = q * scale. 단순 곱셈으로 근사 float를 복구함."""
    return [[q * scale for q in row] for row in quantized]


def dequantize_zeropoint(
    quantized: list[list[int]], scale: float, zero_point: int,
) -> list[list[float]]:
    """w_hat = (q - zero_point) * scale. 비대칭 시프트를 되돌림."""
    return [[(q - zero_point) * scale for q in row] for row in quantized]


def dequantize_per_channel(
    quantized: list[list[int]], scales: list[float],
) -> list[list[float]]:
    """w_hat[i] = q[i] * scale[row_index]. 각 행이 자체 스케일을 사용함."""
    return [
        [q * scales[i] for q in quantized[i]]
        for i in range(len(quantized))
    ]


# === EVALUATION HELPERS ===

def extract_float_weights(params: dict) -> dict[str, list[list[float]]]:
    """모든 파라미터 행렬에서 Value.data를 일반 float 리스트로 추출함."""
    float_weights: dict[str, list[list[float]]] = {}
    for name, matrix in params.items():
        float_weights[name] = [[v.data for v in row] for row in matrix]
    return float_weights


def compute_model_size(float_weights: dict[str, list[list[float]]], bits: int) -> int:
    """주어진 비트 폭에서의 모델 크기를 바이트 단위로 계산함.

    Float32 = 4 bytes/weight, INT8 = 1 byte/weight, INT4 = 0.5 bytes/weight.
    스케일 팩터의 오버헤드는 무시할 수준 (행렬 또는 행당 float 하나).
    """
    n_weights = sum(len(row) for matrix in float_weights.values() for row in matrix)
    return int(n_weights * bits / 8)


def compute_roundtrip_error(
    original: dict[str, list[list[float]]],
    dequantized: dict[str, list[list[float]]],
) -> float:
    """모든 가중치에 걸친 최대 절대 오차: max |w - dequant(quant(w))|.

    이건 최악의 단일 가중치 오차임. 평균 오차는 보통 10-100배 작지만,
    최대 오차가 특정 계산 경로에서 치명적 왜곡이 발생하는지를 결정함.
    """
    max_err = 0.0
    for name in original:
        for orig_row, deq_row in zip(original[name], dequantized[name]):
            for o, d in zip(orig_row, deq_row):
                max_err = max(max_err, abs(o - d))
    return max_err


def evaluate_loss(
    float_params: dict[str, list[list[float]]],
    eval_docs: list[str],
    unique_chars: list[str],
    bos: int,
) -> float:
    """float 순전파를 사용한 평가 문서의 평균 크로스엔트로피 loss."""
    total_loss = 0.0
    total_tokens = 0
    for doc in eval_docs:
        tokens = [bos] + [unique_chars.index(ch) for ch in doc] + [bos]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)
        keys: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
        values: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
        for pos in range(seq_len):
            logits = gpt_forward_float(tokens[pos], pos, keys, values, float_params)
            probs = softmax_float(logits)
            # 크로스엔트로피: -log(p(target))
            p_target = max(probs[tokens[pos + 1]], 1e-10)
            total_loss += -math.log(p_target)
            total_tokens += 1
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def generate_sample(
    float_params: dict[str, list[list[float]]],
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    temperature: float = 0.5,
) -> str:
    """temperature 스케일링 샘플링으로 이름을 하나 생성함."""
    keys: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
    values: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
    token_id = bos
    generated: list[str] = []
    for pos in range(BLOCK_SIZE):
        logits = gpt_forward_float(token_id, pos, keys, values, float_params)
        scaled = [l / temperature for l in logits]
        probs = softmax_float(scaled)
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == bos:
            break
        generated.append(unique_chars[token_id])
    return ''.join(generated)


# === TRAINING AND QUANTIZATION ===

if __name__ == "__main__":
    t_start = time.time()

    # === PHASE 1: TRAIN BASE MODEL ===
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} documents, vocabulary: {VOCAB_SIZE} tokens")

    # 파라미터 초기화
    params: dict[str, list[list[Value]]] = {}
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)
    for li in range(N_LAYER):
        params[f'layer{li}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{li}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{li}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{li}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{li}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{li}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    param_list = [p for matrix in params.values() for row in matrix for p in row]
    print(f"Parameters: {len(param_list):,}\n")

    # Adam 옵티마이저 상태
    m = [0.0] * len(param_list)
    v = [0.0] * len(param_list)

    print("Training base model...")
    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys: list[list[list[Value]]] = [[] for _ in range(N_LAYER)]
        values: list[list[list[Value]]] = [[] for _ in range(N_LAYER)]

        losses = []
        for pos in range(seq_len):
            logits = gpt_forward(tokens[pos], pos, keys, values, params)
            probs = softmax(logits)
            loss_t = -safe_log(probs[tokens[pos + 1]])
            losses.append(loss_t)

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, param in enumerate(param_list):
            m[i] = BETA1 * m[i] + (1 - BETA1) * param.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * param.grad ** 2
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))
            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            param.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    t_train = time.time()
    print(f"\nTraining complete ({t_train - t_start:.1f}s). Final loss: {loss.data:.4f}")

    # === PHASE 2: EXTRACT FLOAT32 WEIGHTS ===
    # 이 시점부터 autograd를 사용하지 않음. 모든 연산이 일반 float로 수행됨.
    # 이건 프로덕션 quantization을 그대로 반영함: 학습된 모델 체크포인트를 받아서
    # 추가 학습 없이 사후 학습 양자화(PTQ)를 적용함.
    print("\n=== Extracting Float32 Weights ===")
    float_weights = extract_float_weights(params)

    # 일관된 loss 비교를 위해 고정된 평가 세트(처음 200개 문서) 사용
    eval_docs = docs[:200]

    # 원본 float32 가중치의 기준 loss
    baseline_loss = evaluate_loss(float_weights, eval_docs, unique_chars, BOS)
    print(f"Float32 baseline loss: {baseline_loss:.4f}")

    # 모든 quantization 변형에서 재현 가능한 생성을 위한 시드 리셋
    random.seed(42)
    baseline_sample = generate_sample(float_weights, unique_chars, BOS, VOCAB_SIZE)

    # === PHASE 3: QUANTIZE TO INT8 (ABSMAX) ===
    print("\n=== INT8 Absmax Quantization ===")
    int8_weights: dict[str, list[list[float]]] = {}
    for name, matrix in float_weights.items():
        q, s = quantize_absmax_int8(matrix)
        int8_weights[name] = dequantize_absmax(q, s)

    int8_loss = evaluate_loss(int8_weights, eval_docs, unique_chars, BOS)
    random.seed(42)
    int8_sample = generate_sample(int8_weights, unique_chars, BOS, VOCAB_SIZE)
    int8_err = compute_roundtrip_error(float_weights, int8_weights)
    print(f"INT8 absmax loss: {int8_loss:.4f} (delta: {(int8_loss - baseline_loss) / baseline_loss * 100:+.1f}%)")

    # === PHASE 4: QUANTIZE TO INT4 (ABSMAX) ===
    print("\n=== INT4 Absmax Quantization ===")
    int4_weights: dict[str, list[list[float]]] = {}
    for name, matrix in float_weights.items():
        q, s = quantize_absmax_int4(matrix)
        int4_weights[name] = dequantize_absmax(q, s)

    int4_loss = evaluate_loss(int4_weights, eval_docs, unique_chars, BOS)
    random.seed(42)
    int4_sample = generate_sample(int4_weights, unique_chars, BOS, VOCAB_SIZE)
    int4_err = compute_roundtrip_error(float_weights, int4_weights)
    print(f"INT4 absmax loss: {int4_loss:.4f} (delta: {(int4_loss - baseline_loss) / baseline_loss * 100:+.1f}%)")

    # === PHASE 5: ZERO-POINT QUANTIZATION (ASYMMETRIC INT8) ===
    # 가중치 분포가 0에서 벗어났을 때 유용함. ReLU 활성화 후나
    # 특정 초기화 방식에서 가중치가 비대칭적으로 분포할 수 있음.
    # Zero-point 매핑이 이 비대칭성을 캡처함.
    print("\n=== INT8 Zero-Point Quantization ===")
    zp_weights: dict[str, list[list[float]]] = {}
    for name, matrix in float_weights.items():
        q, s, zp = quantize_zeropoint_int8(matrix)
        zp_weights[name] = dequantize_zeropoint(q, s, zp)

    zp_loss = evaluate_loss(zp_weights, eval_docs, unique_chars, BOS)
    random.seed(42)
    zp_sample = generate_sample(zp_weights, unique_chars, BOS, VOCAB_SIZE)
    zp_err = compute_roundtrip_error(float_weights, zp_weights)
    print(f"INT8 zero-point loss: {zp_loss:.4f} (delta: {(zp_loss - baseline_loss) / baseline_loss * 100:+.1f}%)")

    # === PHASE 6: PER-CHANNEL INT8 ===
    # Per-tensor quantization은 가장 큰 이상치가 있는 행에 의해 제한됨.
    # Per-channel quantization은 각 출력 채널에 자체 스케일을 부여해서,
    # 하나의 이상치 행이 전체 행렬의 정밀도를 저하시키지 않음.
    print("\n=== INT8 Per-Channel Quantization ===")
    pc_weights: dict[str, list[list[float]]] = {}
    for name, matrix in float_weights.items():
        q, scales = quantize_per_channel_int8(matrix)
        pc_weights[name] = dequantize_per_channel(q, scales)

    pc_loss = evaluate_loss(pc_weights, eval_docs, unique_chars, BOS)
    random.seed(42)
    pc_sample = generate_sample(pc_weights, unique_chars, BOS, VOCAB_SIZE)
    pc_err = compute_roundtrip_error(float_weights, pc_weights)
    print(f"INT8 per-channel loss: {pc_loss:.4f} (delta: {(pc_loss - baseline_loss) / baseline_loss * 100:+.1f}%)")

    # === PHASE 7: COMPARISON TABLE ===
    t_end = time.time()

    size_32 = compute_model_size(float_weights, 32)
    size_8 = compute_model_size(float_weights, 8)
    size_4 = compute_model_size(float_weights, 4)

    print("\n" + "=" * 80)
    print("=== Quantization Results ===")
    print("=" * 80)

    header = f"{'Method':<24} {'Bits':>4} {'Size':>9} {'Loss':>8} {'Delta':>8} {'Max Err':>10} {'Sample':<14}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Float32 (baseline)", 32, size_32, baseline_loss, 0.0, 0.0, baseline_sample),
        ("INT8 absmax", 8, size_8, int8_loss,
         (int8_loss - baseline_loss) / baseline_loss * 100, int8_err, int8_sample),
        ("INT8 per-channel", 8, size_8, pc_loss,
         (pc_loss - baseline_loss) / baseline_loss * 100, pc_err, pc_sample),
        ("INT8 zero-point", 8, size_8, zp_loss,
         (zp_loss - baseline_loss) / baseline_loss * 100, zp_err, zp_sample),
        ("INT4 absmax", 4, size_4, int4_loss,
         (int4_loss - baseline_loss) / baseline_loss * 100, int4_err, int4_sample),
    ]

    for name, bits, size, loss_val, delta, err, sample in rows:
        delta_str = "---" if delta == 0.0 else f"{delta:+.1f}%"
        err_str = "---" if err == 0.0 else f"{err:.6f}"
        # 표시용으로 샘플을 자름
        sample_disp = sample[:12] if len(sample) > 12 else sample
        print(f"{name:<24} {bits:>4} {size:>7,} B {loss_val:>8.4f} {delta_str:>8} {err_str:>10} {sample_disp:<14}")

    print(f"\nCompression ratios: float32->INT8 = {size_32 / size_8:.1f}x, "
          f"float32->INT4 = {size_32 / size_4:.1f}x")

    # 핵심 발견 강조: per-channel이 per-tensor보다 나음
    if pc_loss < int8_loss:
        print("Per-channel INT8 outperforms per-tensor INT8 (lower loss delta).")
    else:
        print("Per-tensor and per-channel INT8 performed comparably on this small model.")

    # === WHY QUANTIZATION WORKS ===
    # 신경망이 가중치 정밀도 손실에 강건한 두 가지 이유:
    # 1. 중복성: 수천 개의 가중치가 각 출력에 기여하므로, 개별 반올림
    #    오차가 평균화됨. 중심극한정리가 작동하는 것임.
    # 2. 가중치 분포가 대략 가우시안이고 분산이 작음.
    #    대부분의 가중치가 quantization 격자가 가장 촘촘한 0 근처에 있음.
    #    이상치 가중치만 큰 반올림 오차를 겪음.
    #
    # INT8은 ~99%의 정보를 보존함 (256 quantization 레벨).
    # INT4가 실용적 한계임 -- 16 레벨은 눈에 띄는 저하를 일으킴.
    # INT2 (4 레벨)는 보통 모델 품질을 완전히 파괴함.
    #
    # 참고: 프로덕션 quantization은 이 스크립트가 생략한 여러 정교한 기법을 추가함:
    # - GPTQ/AWQ는 보정 데이터로 레이어별 출력 재구성 오차를 최소화함
    # - 혼합 정밀도는 민감한 레이어(첫째/마지막)를 더 높은 정밀도로 유지함
    # - 활성화 quantization (가중치뿐 아니라)으로 완전한 정수 추론
    # - 그룹 quantization: per-channel이지만 스케일당 32-128개 원소 그룹
    # - SmoothQuant는 quantization 난이도를 활성화에서 가중치로 이전함

    print(f"\nTotal runtime: {t_end - t_start:.1f}s")
