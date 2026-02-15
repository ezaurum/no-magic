"""
자기회귀 언어 모델을 처음부터 구현함: GPT는 행렬 곱셈, attention, gradient descent만으로
시퀀스에서 다음 문자를 예측하는 법을 배움.
"""
# Reference: This implementation follows the GPT-2 architecture (Radford et al., 2019)
# with pedagogical simplifications: RMSNorm instead of LayerNorm, ReLU instead of GELU,
# no bias terms. Algorithmic flow inspired by Karpathy's microgpt.py but rewritten from
# scratch with comprehensive commenting for educational clarity.
# 교육적 단순화를 적용함: LayerNorm 대신 RMSNorm, GELU 대신 ReLU, bias 없음.
# Karpathy의 microgpt.py에서 알고리즘 흐름을 참고했으나, 교육 목적으로 처음부터 다시 작성하고
# 상세한 주석을 달았음.

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# 모델 아키텍처
N_EMBD = 16         # embedding 차원 (Transformer 논문에서 d_model)
N_HEAD = 4          # attention head 수
N_LAYER = 1         # transformer 블록 수
BLOCK_SIZE = 16     # 컨텍스트 윈도우 크기 (최대 시퀀스 길이)
HEAD_DIM = N_EMBD // N_HEAD  # attention head당 차원 (16/4 = 4)

# 학습 파라미터
LEARNING_RATE = 0.01  # Adam 기본 학습률
BETA1 = 0.85          # Adam 1차 모멘트 감쇠율
BETA2 = 0.99          # Adam 2차 모멘트 감쇠율
EPS_ADAM = 1e-8       # Adam epsilon (0으로 나누는 것을 방지함)
NUM_STEPS = 1000      # 총 학습 스텝 수

# 데이터 파라미터
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 참고: 총 ~4,200개 파라미터임. 실제 GPT는 수십억 개를 가짐. 아키텍처는
# 동일하지만 (attention은 attention임), 이 소규모로는 GPU 클러스터에서 수 주 대신
# CPU에서 수 분 만에 학습할 수 있음.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """학습 데이터를 다운로드하고 파싱함."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    with open(filename, "r") as f:
        # 각 줄이 하나의 문서(이름)임. 공백을 제거하고 빈 줄을 필터링함.
        docs = [line.strip() for line in f if line.strip()]

    return docs


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """역전파 자동 미분을 지원하는 스칼라 값.

    ._children와 ._local_grads를 통해 계산 이력을 추적하며, chain rule을 사용해
    gradient를 계산함. 모든 순전파 연산은 자신의 로컬 미분값(∂out/∂input)을 클로저로
    저장하고, backward()는 계산 그래프를 역위상 순서로 역방향 재생하면서 gradient를
    누적함.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data          # 스칼라 float 값
        self.grad = 0.0           # 누적 gradient (∂Loss/∂self)
        self._children = children # 계산 그래프에서의 부모 Value들
        self._local_grads = local_grads  # 각 child에 대한 ∂self/∂child

    # 산술 연산
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

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # 활성화 함수
    def tanh(self):
        # d(tanh(x))/dx = 1 - tanh(x)^2
        # 표준 활성화 함수임 -- microgpt는 ReLU를 쓰지만, tanh는
        # 다른 스크립트(micrornn, microlora)를 위한 표준 인터페이스의 일부임.
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self):
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        # d(log(x))/dx = 1/x
        # 입력이 이미 클램핑되었다고 가정함 (아래 safe_log 참조)
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        # d(relu(x))/dx = 1 if x > 0 else 0
        # ReLU는 매우 단순함: max(0, x). 양수 입력이면 gradient가 1, 아니면 0임.
        # 최신 transformer는 보통 GELU를 쓰지만, ReLU가 이해하기 쉽고
        # 질적으로 비슷한 결과를 냄.
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """역전파 자동 미분으로 gradient를 계산함.

        계산 그래프의 위상 정렬을 만든 뒤, chain rule로 gradient를 역방향 전파함.
        합성 함수 f(g(h(x)))에서, chain rule에 의해 df/dx = (df/dg) * (dg/dh) * (dh/dx)임.
        위상 정렬이 df/dg를 df/dh에 필요하기 전에 먼저 계산하도록 보장함.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 시드: loss의 자기 자신에 대한 gradient는 1임
        self.grad = 1.0

        # 역위상 순서: gradient가 출력에서 입력 방향으로 흐름
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain rule: ∂Loss/∂child += ∂Loss/∂v * ∂v/∂child
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# 이 Value 클래스는 표준 인터페이스를 정확히 따름.
# 전체 스펙은 docs/autograd-interface.md를 참조.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """가우시안 노이즈로 가중치 행렬을 초기화함.

    표준편차 0.08은 이 작은 모델에 경험적으로 선택된 값임 -- 더 큰 모델은
    보통 std = 1/sqrt(d_in) (Xavier/Glorot 초기화)을 써서 깊은 층을 지날 때
    활성화값이 폭발하거나 소멸하지 않게 함. 1개 층만 있으면 초기화가 덜 중요함.
    """
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_parameters():
    """모든 모델 파라미터를 초기화함: embedding, attention, MLP 가중치.

    사람이 읽기 쉬운 이름으로 키가 지정된 dict를 반환함. 이것이 "state_dict"임 --
    학습된 모델의 완전한 명세. 이 dict를 저장하면 모델을 저장한 것임.
    """
    params = {}

    # 토큰 및 위치 embedding
    # wte: [vocab_size, n_embd] - 토큰 ID를 벡터로 매핑함
    # wpe: [block_size, n_embd] - 위치(0..15)를 벡터로 매핑함
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    # 레이어별 가중치 (1개 층만 있지만, 패턴은 일반화 가능함)
    for layer_idx in range(N_LAYER):
        # Attention 가중치 (Q, K, V 프로젝션 및 출력 프로젝션)
        # 모두 정방 [n_embd, n_embd] 행렬임
        params[f'layer{layer_idx}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)

        # MLP 가중치 (확장 계수 4인 2층 feedforward 네트워크)
        # fc1: [n_embd, 4*n_embd] - 확장, fc2: [4*n_embd, n_embd] - 축소
        # 4배 확장은 GPT 관례임 -- residual stream 너비를 늘리지 않으면서
        # MLP에 attention 출력을 처리할 더 많은 용량을 줌.
        params[f'layer{layer_idx}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)

    # 언어 모델 헤드: 최종 hidden state를 어휘 logits으로 프로젝션함
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    return params


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """행렬-벡터 곱셈: y = W @ x (bias 없음).

    가중치 행렬 W의 형태가 [n_out, n_in]이고 입력 벡터 x의 형태가 [n_in]일 때,
    출력 y의 형태는 [n_out]이며 각 원소 y[i] = sum_j W[i,j] * x[j]임.
    이것이 신경망의 기본 연산임: 모든 층은 linear() 뒤에 비선형 활성화가 오는 것임.
    """
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """수치적으로 안정적인 softmax: logits을 확률로 변환함.

    Softmax는 이동 불변임: 어떤 c에 대해서든 softmax(x) = softmax(x - c)임.
    exp() 전에 max(x)를 빼서 오버플로를 방지함. 이렇게 안 하면 큰 logits(>700)이
    exp()에서 inf를 반환하여 계산이 깨짐.

    Math: softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    Stable: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """Root Mean Square 정규화: 벡터를 단위 RMS 크기로 스케일링함.

    RMSNorm은 평균 중심화나 학습 가능한 affine 파라미터가 없는 LayerNorm임.
    연산이 적고, 파라미터가 적으며, 경험적으로 동일하게 잘 작동함 (LLaMA,
    Gemma 등 최근 아키텍처에서 사용됨).

    Math: RMSNorm(x) = x / sqrt(mean(x^2) + eps)
    epsilon (1e-5)은 x가 모두 0일 때 0으로 나누는 것을 방지함.
    """
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """loss 계산에서 수치 안정성을 위한 클리핑된 로그.

    log(0)은 -inf를 반환하여 gradient 역전파를 깨뜨리므로 이를 방지함.
    1e-10으로 클램핑하면 log(1e-10) ≈ -23이 되어 유한하고 gradient 정보를
    보존함. 이렇게 안 하면, 학습 초기에 발생할 수 있는 단 하나의 0 확률이
    전체 gradient를 죽임.

    중요: gradient가 계산 그래프를 통해 역류할 수 있도록 `prob`을 자식 노드로
    유지해야 함. 연결이 끊긴 Value(clamped)를 만들면 gradient 경로가 끊겨서
    모델이 학습하지 못함.
    """
    clamped = max(prob.data, 1e-10)
    # prob을 자식으로 유지하면서 log 노드를 수동으로 만들어 그래프를 보존함.
    # d(log(x))/dx = 1/x, 안정성을 위해 클램핑된 값에서 평가함.
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === GPT FORWARD PASS ===

def gpt_forward(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
    params: dict,
) -> list[Value]:
    """GPT 모델의 단일 토큰 순전파.

    이 함수는 위치 `pos_id`에서 하나의 토큰을 처리하고 어휘에 대한 logits을
    반환함. keys와 values 리스트는 KV 캐시를 누적함 -- 과거 모든 토큰의
    key/value 프로젝션의 실행 이력이며, 명시적 마스크 행렬 없이 causal attention을
    구현할 수 있게 해줌.

    Args:
        token_id: 입력 토큰을 식별하는 [0, vocab_size-1] 범위의 정수
        pos_id: 시퀀스 내 위치를 나타내는 [0, block_size-1] 범위의 정수
        keys: key용 KV 캐시, 형태 [n_layer][seq_len][n_embd]
        values: value용 KV 캐시, 형태 [n_layer][seq_len][n_embd]
        params: 모델 가중치 행렬

    Returns:
        어휘에 대한 logits (정규화되지 않은 로그 확률), 길이 vocab_size
    """
    # -- Embedding 층 --
    # 이 토큰과 위치에 대한 학습된 벡터를 조회한 뒤 더함.
    # 이것이 GPT 입력 표현임: tok_emb는 "무엇"(토큰)을 인코딩하고,
    # pos_emb는 "어디"(시퀀스 내 위치)를 인코딩함.
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    # transformer 블록에 넣기 전에 embedding을 정규화함
    x = rmsnorm(x)

    # -- Transformer 층 --
    for layer_idx in range(N_LAYER):
        # Residual connection 패턴: x_new = x + f(x)
        # 이 "고속도로"가 gradient를 attention이나 MLP를 거치지 않고 모델을 통해
        # 직접 역방향으로 흐르게 해서 vanishing gradient를 방지함.
        x_residual = x

        # Pre-norm: attention 전에 정규화함 (최신 아키텍처는 post-norm 대신
        # 이 방식을 씀, 깊은 모델에서 학습이 안정화되기 때문임)
        x = rmsnorm(x)

        # -- Multi-head self-attention --
        # 입력을 query, key, value로 프로젝션함
        q = linear(x, params[f'layer{layer_idx}.attn_wq'])
        k = linear(x, params[f'layer{layer_idx}.attn_wk'])
        v = linear(x, params[f'layer{layer_idx}.attn_wv'])

        # 이 레이어의 캐시에 k, v를 추가함. KV 캐시를 점진적으로 구축함:
        # 위치 t에서, keys[layer_idx]는 [k_0, k_1, ..., k_t]를 포함함.
        keys[layer_idx].append(k)
        values[layer_idx].append(v)

        # 각 attention head를 독립적으로 처리한 뒤 출력을 연결함
        x_attn = []
        for head in range(N_HEAD):
            head_start = head * HEAD_DIM

            # 이 head의 q/k/v 벡터 부분을 슬라이스함
            q_head = q[head_start : head_start + HEAD_DIM]
            k_head = [k_t[head_start : head_start + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[head_start : head_start + HEAD_DIM] for v_t in values[layer_idx]]

            # Attention 스코어 계산: 각 과거 토큰에 얼마나 주목할지 결정함
            # 공식: score(q, k_t) = (q · k_t) / sqrt(d_head)
            # sqrt(d_head) 스케일링은 차원이 커질 때 스코어가 너무 커지는 것을
            # 방지함 (그러면 softmax가 포화됨).
            attn_logits = [
                sum(q_head[j] * k_head[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]

            # 스코어를 softmax로 확률로 변환함
            attn_weights = softmax(attn_logits)

            # value의 가중 합: output[j] = sum_t attn_weights[t] * v[t][j]
            # 이것이 "attention" 메커니즘임: 모든 과거 토큰을 (value 벡터를 통해)
            # 보고 각각의 관련성(attention 가중치)으로 가중함.
            head_output = [
                sum(attn_weights[t] * v_head[t][j] for t in range(len(v_head)))
                for j in range(HEAD_DIM)
            ]

            x_attn.extend(head_output)

        # 참고: KV 캐싱이 명시적 마스크 없이 causal masking을 제공하는 이유 --
        # 위치 t에서, keys[layer_idx]는 위치 0..t의 key만 포함하므로,
        # attention 스코어 루프(range(len(k_head)))가 자연스럽게 미래 토큰을 제외함.
        # 이 점진적 구축은 배치 설정에서 하삼각 마스크를 적용하는 것과 동일하지만,
        # 자기회귀 생성에는 더 효율적임.

        # 연결된 head 출력을 residual 차원으로 다시 프로젝션함
        x = linear(x_attn, params[f'layer{layer_idx}.attn_wo'])

        # 첫 번째 residual connection (attention 주변)
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        # -- MLP (feedforward 네트워크) --
        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])  # 확장
        x = [xi.relu() for xi in x]                         # 비선형성
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])  # 축소

        # 두 번째 residual connection (MLP 주변)
        x = [a + b for a, b in zip(x, x_residual)]

    # -- 출력 층 --
    # 최종 hidden state를 어휘 logits으로 프로젝션함
    logits = linear(x, params['lm_head'])
    return logits


# === TRAINING LOOP ===

if __name__ == "__main__":
    # -- 어휘 및 데이터 준비 --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    # 코퍼스의 고유 문자로 어휘를 구축함
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)  # 시퀀스 시작 토큰 (문자 집합에 추가됨)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} documents")
    print(f"Vocabulary size: {VOCAB_SIZE} (characters + BOS token)")

    # 어휘 크기를 알게 된 후 파라미터를 초기화함
    params = init_parameters()

    # optimizer 관리를 위해 모든 파라미터를 단일 리스트로 평탄화함
    param_list = [p for matrix in params.values() for row in matrix for p in row]
    print(f"Parameters: {len(param_list):,}\n")

    # -- Adam optimizer 상태 초기화 --
    # m: 1차 모멘트 (모멘텀), v: 2차 모멘트 (분산)
    # gradient 이력을 기반으로 각 가중치에 개별적으로 학습률을 적응시키는
    # 파라미터별 이동 평균임.
    m = [0.0] * len(param_list)
    v = [0.0] * len(param_list)

    # -- 학습 --
    print("Training...")
    for step in range(NUM_STEPS):
        # 데이터셋을 순환함 (셔플링으로 본질적으로 SGD임)
        doc = docs[step % len(docs)]

        # 토큰화: 문서를 BOS 마커가 포함된 정수 시퀀스로 변환함
        # 형식: [BOS, char_0, char_1, ..., char_n, BOS]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]

        # block_size (컨텍스트 윈도우 한계)로 잘라냄
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        # 이 시퀀스에 대한 KV 캐시를 초기화함 (각 문서마다 새로 시작)
        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]

        # 시퀀스에 걸쳐 loss를 계산함 (각 위치에서 cross-entropy)
        losses = []
        for pos in range(seq_len):
            input_token = tokens[pos]
            target_token = tokens[pos + 1]

            # 순전파
            logits = gpt_forward(input_token, pos, keys, values, params)

            # logits을 확률로 변환함
            probs = softmax(logits)

            # 음의 로그 우도 loss: -log(p(target))
            # 분류를 위한 cross-entropy loss임. 모델이 실제 다음 토큰에
            # 높은 확률을 부여하길 원함.
            loss_t = -safe_log(probs[target_token])
            losses.append(loss_t)

        # 시퀀스에 대한 평균 loss (문서 길이에 무관하게 loss 스케일을 맞춤)
        loss = (1.0 / seq_len) * sum(losses)

        # -- 역전파 --
        loss.backward()

        # -- Adam optimizer 스텝 --
        # 선형 학습률 감쇠: lr_t = lr_0 * (1 - t/T)
        # 이 "학습률 웜다운"은 최적점 근처에서 loss 표면이 날카로워질 때 과도한
        # 이동을 방지함. 감쇠 없이는 고정된 스텝 크기가 최솟값 주변에서 왔다 갔다
        # 하게 만들어 수렴하지 못할 수 있음.
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)

        for i, param in enumerate(param_list):
            # Adam 업데이트 규칙:
            # m_t = β1*m_{t-1} + (1-β1)*g_t         (모멘텀)
            # v_t = β2*v_{t-1} + (1-β2)*g_t^2       (분산)
            # θ_t = θ_{t-1} - lr * m_hat / (sqrt(v_hat) + ε)
            m[i] = BETA1 * m[i] + (1 - BETA1) * param.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * param.grad ** 2

            # 편향 보정: m과 v가 0으로 초기화되어 초기 스텝에서 0쪽으로 편향됨.
            # (1 - β^t)로 나누어 이를 보정함.
            # 편향 보정이 없으면 초기 업데이트가 너무 작아짐.
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))

            # 파라미터 업데이트
            # epsilon (1e-8)은 v_hat이 매우 작을 때 0으로 나누는 것을 방지함
            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)

            # 다음 반복을 위해 gradient를 0으로 초기화함
            param.grad = 0.0

        # 진행 상황 출력
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS:>4} | loss: {loss.data:.4f}")

    print(f"\nTraining complete. Final loss: {loss.data:.4f}\n")

    # === INFERENCE ===
    # temperature 스케일링 샘플링으로 학습된 모델에서 새 샘플을 생성함.
    # Temperature는 무작위성을 제어함: 낮을수록 더 결정적, 높을수록 더 무작위.
    TEMPERATURE = 0.5
    NUM_SAMPLES = 20

    print(f"Generating {NUM_SAMPLES} samples (temperature={TEMPERATURE}):\n")

    for sample_idx in range(NUM_SAMPLES):
        # 각 샘플마다 새로운 KV 캐시
        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]

        # BOS 토큰으로 시작함
        token_id = BOS
        generated = []

        for pos in range(BLOCK_SIZE):
            # 순전파
            logits = gpt_forward(token_id, pos, keys, values, params)

            # Temperature 스케일링: softmax 전에 logits을 temperature로 나눔
            # 확률 분포를 날카롭게(T < 1) 또는 평평하게(T > 1) 만듦.
            # 낮은 temperature는 모델을 더 확신하게 만들고 (높은 확률 토큰을 선택),
            # 높은 temperature는 더 탐색적으로 만듦 (더 균등하게 샘플링).
            scaled_logits = [logit / TEMPERATURE for logit in logits]
            probs = softmax(scaled_logits)

            # 확률 분포에서 다음 토큰을 샘플링함
            # random.choices는 확률을 샘플링 가중치로 사용함
            token_id = random.choices(
                range(VOCAB_SIZE),
                weights=[p.data for p in probs]
            )[0]

            # BOS (시퀀스 종료 마커)를 만나면 중단함
            if token_id == BOS:
                break

            generated.append(unique_chars[token_id])

        # 생성된 이름을 출력함
        print(f"  {sample_idx + 1:>2}. {''.join(generated)}")
