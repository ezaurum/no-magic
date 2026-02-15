"""
Low-Rank Adaptation (LoRA): 고정된 language model에 작은 학습 가능 행렬을 주입해서 fine-tuning하는 방법
— weight 업데이트가 low-dimensional 부분공간에 존재함을 증명함.
"""
# Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).
# https://arxiv.org/abs/2106.09685
# microgpt 패턴(Radford et al., 2019)을 재사용하며 교육적 목적으로 단순화함:
# RMSNorm, ReLU, bias 없음. LoRA adapter를 Q와 V projection에 적용함.

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# 모델 아키텍처 — microgpt와 동일해서 직접 비교 가능함
N_EMBD = 16         # embedding 차원 (d_model)
N_HEAD = 4          # attention head 수
N_LAYER = 1         # transformer 블록 수
BLOCK_SIZE = 16     # context window 길이
HEAD_DIM = N_EMBD // N_HEAD  # head당 4차원

# LoRA hyperparameter
LORA_RANK = 2       # adaptation 행렬의 rank (r << d_model)
# rank 2는 각 adapter 쌍이 weight 행렬에 rank-2 perturbation을 기여한다는 뜻임.
# 실제 LoRA는 보통 r=4..64를 사용함. d_model=16에서는 r=2만으로도 의미 있는
# 구조를 포착하면서 데모용으로 parameter 수를 눈에 띄게 작게 유지함.

# 학습 — base model
BASE_LR = 0.01
BASE_STEPS = 800
# 학습 — LoRA adaptation
LORA_LR = 0.01
LORA_STEPS = 500

# 공유 optimizer 상수
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8

# 데이터
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 참고: base parameter ~4,200개. 실제 모델은 수십억 개임. LoRA의 가치는
# 스케일이 커질수록 극적으로 드러남: 7B 모델을 r=16으로 adaptation하면 ~0.1%의 parameter만 학습함.
# 우리의 toy 스케일에서는 비율이 덜 극적이지만 메커니즘은 동일함.


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


# --- AUTOGRAD IN THIS SCRIPT ---
# 이 Value 클래스는 canonical interface를 그대로 따름.
# 전체 명세는 docs/autograd-interface.md 참조.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """weight 행렬을 ~ N(0, std)로 초기화함. 표준편차 0.08은 이 tiny model에 대해
    경험적으로 튜닝된 값임; 큰 모델은 Xavier/Glorot 스케일링(std = 1/sqrt(d_in))을 사용함."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def make_lora_A(nrows: int, ncols: int) -> list[list[Value]]:
    """LoRA A 행렬을 ~ N(0, 0.02)로 초기화함.
    작은 랜덤 초기화로 두 adapter(A와 B)의 대칭성을 깨뜨림. B가 0으로 시작하므로
    초기 LoRA 기여는 A의 값과 무관하게 A @ 0 = 0임 — 하지만
    B가 학습을 시작하면 A의 랜덤 방향이 다양한 gradient 신호를 제공함."""
    return [[Value(random.gauss(0, 0.02)) for _ in range(ncols)] for _ in range(nrows)]


def make_lora_B(nrows: int, ncols: int) -> list[list[Value]]:
    """LoRA B 행렬을 0으로 초기화함.
    # Math: W_adapted = W_frozen + A @ B
    # 초기 상태: A @ B = A @ 0 = 0이므로 adapted model은 base model과 동일함.
    # 이것이 핵심임: LoRA가 pretrained 솔루션에서 시작해서 작은 perturbation을
    # 만든다는 뜻이며, base model이 학습한 것을 즉시 파괴하는 랜덤 오프셋에서
    # 시작하는 게 아님."""
    return [[Value(0.0) for _ in range(ncols)] for _ in range(nrows)]


def init_parameters(vocab_size: int) -> dict[str, list[list[Value]]]:
    """모든 base model parameter를 초기화함: embedding, attention, MLP, LM head."""
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


def init_lora_adapters() -> dict[str, list[list[Value]]]:
    """Q와 V attention projection에 대한 LoRA adapter 행렬을 생성함.

    왜 K나 O가 아니라 Q와 V인가? 원래 LoRA 논문(Hu et al., 2021)에서 Q와 V projection을
    adaptation하는 것이 parameter당 가장 많은 task 관련 정보를 포착한다고 밝힘.
    직관적으로: Q는 "무엇을 찾을지"를, V는 "무엇을 추출할지"를 제어하며 — 둘 다
    task에 매우 특화됨. K("무엇을 알릴지")와 O("어떻게 합칠지")는 task 간 변화가 적음.
    실제 LoRA는 최대 품질을 위해 네 가지 모두 adaptation하는 경우가 많음.
    """
    adapters: dict[str, list[list[Value]]] = {}

    for layer_idx in range(N_LAYER):
        # Q adapter: A는 (N_EMBD, LORA_RANK), B는 (LORA_RANK, N_EMBD)
        # Math: Q_adapted = W_q @ x + A_q @ (B_q @ x)
        #   여기서 A_q @ B_q는 W_q에 대한 rank-r perturbation임
        adapters[f'layer{layer_idx}.lora_q_A'] = make_lora_A(N_EMBD, LORA_RANK)
        adapters[f'layer{layer_idx}.lora_q_B'] = make_lora_B(LORA_RANK, N_EMBD)

        # V adapter: 동일한 구조
        adapters[f'layer{layer_idx}.lora_v_A'] = make_lora_A(N_EMBD, LORA_RANK)
        adapters[f'layer{layer_idx}.lora_v_B'] = make_lora_B(LORA_RANK, N_EMBD)

    return adapters


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """행렬-벡터 곱: y = W @ x. W의 shape이 [n_out, n_in]이고 x의 shape이
    [n_in]일 때, 출력 y의 shape은 [n_out]이며 y[i] = sum_j W[i,j] * x[j]."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def lora_linear(
    x: list[Value],
    w_frozen: list[list[Value]],
    lora_A: list[list[Value]],
    lora_B: list[list[Value]],
) -> list[Value]:
    """LoRA가 적용된 linear 연산: y = W_frozen @ x + A @ (B @ x).

    Math: W_adapted = W_frozen + A @ B  (하지만 이걸 명시적으로 만들지는 않음)
    대신 이렇게 계산함: base_out = W_frozen @ x     (shape: d_out)
                        lora_mid = B @ x             (shape: r)     -- low rank로 projection
                        lora_out = A @ lora_mid      (shape: d_out) -- 다시 원래 차원으로 projection
                        result   = base_out + lora_out

    low-rank bottleneck(r=2)은 adaptation이 출력을 2차원 부분공간에서만 수정할 수
    있다는 뜻임. 이건 한계가 아니라 핵심 통찰임:
    fine-tuning weight 업데이트는 경험적으로 low-rank이므로, rank-2 perturbation이
    유용한 adaptation 신호 대부분을 포착함.

    참고: 실제 LoRA는 adapter 출력에 alpha/r 스케일링 팩터를 적용함.
    r=2에서는 그 효과가 learning rate에 흡수되므로 여기서는 생략함.
    """
    base_out = linear(x, w_frozen)
    # B가 d_in에서 r로 projection (압축 단계)
    lora_hidden = linear(x, lora_B)
    # A가 r에서 d_out으로 projection (확장 단계)
    lora_out = linear(lora_hidden, lora_A)
    return [b + l for b, l in zip(base_out, lora_out)]


def softmax(logits: list[Value]) -> list[Value]:
    """안정적인 softmax: overflow 방지를 위해 exp 전에 max를 뺌.
    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))"""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMS 정규화: x / sqrt(mean(x^2) + eps).
    LayerNorm보다 단순함 (평균 중심화 없음, 학습 가능한 affine 없음). LLaMA, Gemma에서 사용됨."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """수치 안정성을 위한 클리핑된 log. log(0) = -inf를 방지해서 gradient 전파가
    깨지지 않게 함. prob을 child로 하여 노드를 수동으로 생성하므로
    gradient가 computation graph를 통해 역전파됨 (clamping으로 끊기지 않음)."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === GPT FORWARD PASS ===

def gpt_forward(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
    lora: dict[str, list[list[Value]]] | None = None,
) -> list[Value]:
    """단일 토큰 forward pass. lora가 제공되면 Q와 V projection이
    LoRA가 적용된 linear 연산을 사용함; 나머지 weight는 모두 고정 상태임.

    핵심 통찰: forward pass는 LoRA 활성화 여부와 관계없이 구조적으로 동일함.
    유일한 차이는 Q와 V 계산이 linear() 대신 lora_linear()을 거친다는 것임.
    이 조합 가능성이 LoRA가 실용적인 이유 — 모델 아키텍처 변경 없이
    선택된 weight 적용만 수정하면 됨.
    """
    # Embedding: 토큰 identity + positional encoding
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)

        # Q와 V는 가능할 때 LoRA adapter를 사용함; K와 O는 항상 base만 사용함.
        if lora is not None:
            q = lora_linear(
                x,
                params[f'layer{layer_idx}.attn_wq'],
                lora[f'layer{layer_idx}.lora_q_A'],
                lora[f'layer{layer_idx}.lora_q_B'],
            )
            v_proj = lora_linear(
                x,
                params[f'layer{layer_idx}.attn_wv'],
                lora[f'layer{layer_idx}.lora_v_A'],
                lora[f'layer{layer_idx}.lora_v_B'],
            )
        else:
            q = linear(x, params[f'layer{layer_idx}.attn_wq'])
            v_proj = linear(x, params[f'layer{layer_idx}.attn_wv'])

        k = linear(x, params[f'layer{layer_idx}.attn_wk'])

        keys[layer_idx].append(k)
        values[layer_idx].append(v_proj)

        # Multi-head attention: 각 head가 HEAD_DIM 슬라이스에서 동작함
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

        # MLP 블록: 4배 확장, ReLU, 축소
        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === OPTIMIZER ===

def adam_step(
    param_list: list[Value],
    m_state: list[float],
    v_state: list[float],
    step: int,
    lr: float,
) -> None:
    """bias correction과 linear LR decay가 적용된 Adam 업데이트 한 스텝.

    Adam은 parameter별 momentum(m)과 variance(v) 추정값을 유지함.
    bias correction은 m과 v의 0 초기화를 보상해서,
    그렇지 않으면 초기 업데이트가 너무 작아지는 문제를 해결함.
    """
    lr_t = lr * (1 - step / max(step + 1, 1))
    for i, param in enumerate(param_list):
        m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * param.grad
        v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * param.grad ** 2
        m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
        v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
        param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
        param.grad = 0.0


def flatten_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """parameter dict에서 모든 Value 객체를 flat list로 모음."""
    return [p for matrix in params.values() for row in matrix for p in row]


# === EVALUATION ===

def evaluate_loss(
    docs: list[str],
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    params: dict[str, list[list[Value]]],
    lora: dict[str, list[list[Value]]] | None = None,
    num_samples: int = 50,
) -> float:
    """문서 샘플에 대한 평균 cross-entropy loss를 계산함.
    효율성을 위해 .data만 사용함 (gradient 추적 없음)."""
    total_loss = 0.0
    total_tokens = 0
    for idx in range(min(num_samples, len(docs))):
        doc = docs[idx]
        tokens = [bos] + [unique_chars.index(ch) for ch in doc] + [bos]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        for pos in range(seq_len):
            logits = gpt_forward(tokens[pos], pos, keys, vals, params, lora)
            probs = softmax(logits)
            prob_target = max(probs[tokens[pos + 1]].data, 1e-10)
            total_loss += -math.log(prob_target)
            total_tokens += 1
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def generate_names(
    params: dict[str, list[list[Value]]],
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    lora: dict[str, list[list[Value]]] | None = None,
    num_samples: int = 5,
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
            logits = gpt_forward(token_id, pos, keys, vals, params, lora)
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
    # -- 데이터 로드 및 분할 --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    print(f"Loaded {len(docs)} documents")

    # 첫 글자 기준 분할: A-M은 base 학습용, N-Z는 LoRA adaptation용.
    # 이렇게 하면 깔끔한 분포 변화가 생김 — 두 반쪽은 서로 다른 문자 빈도 분포를 가짐
    # (예: N-Z 이름은 n, s, t, r 글자가 더 많음).
    # LoRA는 처음부터 재학습하지 않고 모델의 학습된 문자 통계를 적응시켜야 함.
    base_docs = [d for d in docs if d[0].upper() <= 'M']
    lora_docs = [d for d in docs if d[0].upper() > 'M']
    random.shuffle(base_docs)
    random.shuffle(lora_docs)

    print(f"Base training set: {len(base_docs)} names (A-M)")
    print(f"LoRA adaptation set: {len(lora_docs)} names (N-Z)")

    # 전체 코퍼스에서 어휘를 구축함 (두 분할 모두 같은 문자 집합을 공유함)
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    # === Phase A: Base Model Training ===
    print("\n=== Phase A: Base Model Training ===")
    params = init_parameters(VOCAB_SIZE)
    base_param_list = flatten_params(params)
    print(f"Parameters: {len(base_param_list):,}")

    m_base = [0.0] * len(base_param_list)
    v_base = [0.0] * len(base_param_list)

    for step in range(BASE_STEPS):
        doc = base_docs[step % len(base_docs)]
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

        # linear LR decay로 최적점 근처에서 loss landscape가 날카로워질 때 overshooting을 방지함
        lr_t = BASE_LR * (1 - step / BASE_STEPS)
        for i, p in enumerate(base_param_list):
            m_base[i] = BETA1 * m_base[i] + (1 - BETA1) * p.grad
            v_base[i] = BETA2 * v_base[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_base[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_base[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{BASE_STEPS} | loss: {loss.data:.4f}")

    print(f"Base training complete. Final loss: {loss.data:.4f}")

    # === Phase B: LoRA Adaptation ===
    print("\n=== Phase B: LoRA Adaptation ===")

    lora_adapters = init_lora_adapters()
    lora_param_list = flatten_params(lora_adapters)

    print(f"Base parameters (frozen): {len(base_param_list):,}")
    print(f"LoRA parameters (trainable): {len(lora_param_list):,}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"Adapted matrices: Q, V projections")

    m_lora = [0.0] * len(lora_param_list)
    v_lora = [0.0] * len(lora_param_list)

    for step in range(LORA_STEPS):
        doc = lora_docs[step % len(lora_docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses = []
        for pos in range(seq_len):
            # forward pass에서 LoRA가 적용된 Q와 V projection을 사용함
            logits = gpt_forward(tokens[pos], pos, keys, vals, params, lora_adapters)
            probs = softmax(logits)
            losses.append(-safe_log(probs[tokens[pos + 1]]))

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        # base model 고정: backward 후 모든 base parameter gradient를 0으로 만듦.
        # backward()는 고정된 weight를 포함한 전체 그래프에 gradient를 전파함.
        # 여기서 그 gradient를 버려서 LoRA parameter만 업데이트되게 함.
        # 이것이 LoRA의 핵심 메커니즘임: pretrained 지식은 W_frozen에 보존되고
        # adaptation 신호는 오직 A와 B를 통해서만 흐름.
        for p in base_param_list:
            p.grad = 0.0

        # LoRA parameter만 업데이트
        lr_t = LORA_LR * (1 - step / LORA_STEPS)
        for i, p in enumerate(lora_param_list):
            m_lora[i] = BETA1 * m_lora[i] + (1 - BETA1) * p.grad
            v_lora[i] = BETA2 * v_lora[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_lora[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_lora[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{LORA_STEPS} | loss: {loss.data:.4f}")

    print(f"LoRA training complete. Final loss: {loss.data:.4f}")

    # === Results ===
    print("\n=== Results ===")

    pct = 100 * len(lora_param_list) / len(base_param_list)
    print(f"Trainable params \u2014 Full fine-tune: {len(base_param_list):,} | "
          f"LoRA: {len(lora_param_list):,} ({pct:.1f}%)")

    # base model에서 생성 (LoRA adapter 없음)
    print("\nGenerating from BASE model (trained on A-M names):")
    base_names = generate_names(params, unique_chars, BOS, VOCAB_SIZE, num_samples=5)
    for i, name in enumerate(base_names):
        print(f"  {i + 1}. {name}")

    # LoRA가 적용된 모델에서 생성
    print("\nGenerating from LoRA-ADAPTED model (adapted to N-Z names):")
    lora_names = generate_names(
        params, unique_chars, BOS, VOCAB_SIZE, lora=lora_adapters, num_samples=5
    )
    for i, name in enumerate(lora_names):
        print(f"  {i + 1}. {name}")

    # 교차 평가: 두 분할에 대해 두 모델의 loss를 측정함.
    # LoRA가 제대로 작동하면:
    #   - base model은 A-M(학습 데이터)에서 잘 하고, N-Z에서는 못 해야 함
    #   - LoRA adapted model은 N-Z에서 개선되면서 A-M에서는 크게 나빠지지 않아야 함
    #     (W_frozen이 A-M 지식을 보존하고 A@B는 작은 perturbation만 추가하므로)
    loss_base_am = evaluate_loss(base_docs, unique_chars, BOS, VOCAB_SIZE, params)
    loss_base_nz = evaluate_loss(lora_docs, unique_chars, BOS, VOCAB_SIZE, params)
    loss_lora_am = evaluate_loss(
        base_docs, unique_chars, BOS, VOCAB_SIZE, params, lora_adapters
    )
    loss_lora_nz = evaluate_loss(
        lora_docs, unique_chars, BOS, VOCAB_SIZE, params, lora_adapters
    )

    print(f"\nLoss on A-M split \u2014 Base: {loss_base_am:.2f} | LoRA-adapted: {loss_lora_am:.2f}")
    print(f"Loss on N-Z split \u2014 Base: {loss_base_nz:.2f} | LoRA-adapted: {loss_lora_nz:.2f}")
