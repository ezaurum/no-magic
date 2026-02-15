"""
attention이 모든 걸 지배하기 전 -- 시퀀스를 recurrence로 모델링하던 방식과,
gating이 RNN을 실제로 작동하게 만든 돌파구였던 이유.
"""
# Reference: Vanilla RNN dates to the 1980s (Rumelhart et al.). GRU (Gated Recurrent Unit)
# introduced by Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for
# Statistical Machine Translation" (2014). 이 구현은 같은 character-level 언어 모델링 태스크에서
# 둘을 나란히 비교해서 gating이 왜 중요한지 보여줌.

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

N_HIDDEN = 32       # hidden state 차원 (7분 런타임에 맞춘 소형 크기)
SEQ_LEN = 16        # 최대 시퀀스 길이
LEARNING_RATE = 0.1   # SGD learning rate — microgpt의 Adam보다 10배 높음. plain
                      # SGD는 adaptive rate가 없어서 훨씬 큰 step이 필요함
NUM_STEPS = 3000    # 모델당 학습 스텝 수 (vanilla RNN 3000, GRU 3000)
TRAIN_SIZE = 200    # 작은 학습 서브셋으로 3000 스텝 동안 각 이름이 ~15번 보임

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 참고: 모델당 ~800개 파라미터 (vanilla RNN과 GRU는 비슷한 크기임).
# 프로덕션 RNN은 수백만 개였음. 아키텍처는 정확하고, 이건 교육용 toy scale임.


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

    ._children와 ._local_grads를 통해 연산 이력을 추적하여 chain rule로
    gradient를 계산함. 모든 forward 연산은 로컬 도함수(∂out/∂input)를 클로저로
    저장하고, backward()가 연산 그래프를 역방향 위상 정렬 순서로 재생하면서
    gradient를 누적함.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data          # 스칼라 float 값
        self.grad = 0.0           # 누적된 gradient (∂Loss/∂self)
        self._children = children # 연산 그래프에서의 부모 Value들
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
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def sigmoid(self):
        # sigmoid(x) = 1 / (1 + exp(-x))
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        # GRU의 gating 활성화 함수: [0,1] 범위의 값이 "forget"과 "update" 가중치 역할을 함.
        # sigmoid(x) ≈ 0이면 gate가 정보를 차단하고,
        # ≈ 1이면 gate가 정보를 통과시킴.
        s = 1.0 / (1.0 + math.exp(-self.data))
        return Value(s, (self,), (s * (1 - s),))

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
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """reverse-mode 자동 미분으로 gradient를 계산함.

        연산 그래프의 위상 정렬을 만들고, chain rule로 gradient를 역전파함.
        합성 함수 f(g(h(x)))에서 chain rule은 df/dx = (df/dg) * (dg/dh) * (dh/dx)임.
        위상 정렬이 df/dg를 df/dh에 필요하기 전에 먼저 계산되도록 보장함.
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

        # 시드: loss의 자기 자신에 대한 gradient는 1
        self.grad = 1.0

        # 역방향 위상 순서: gradient가 출력에서 입력으로 흐름
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain rule: ∂Loss/∂child += ∂Loss/∂v * ∂v/∂child
                child.grad += local_grad * v.grad


# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# 이 Value 클래스는 표준 인터페이스(docs/autograd-interface.md 참조)를 따르며
# 다음이 추가됨:
# - sigmoid(): GRU gating에 필요 (z_t와 r_t 계산)
# 기본 연산 (add, mul, tanh, exp, relu, pow, backward)은
# 표준 스펙과 동일함.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """가우시안 노이즈로 가중치 행렬을 초기화함.

    표준편차 0.08은 이 소형 모델에 맞게 경험적으로 선택됨.
    더 큰 모델은 보통 std = 1/sqrt(d_in) (Xavier/Glorot 초기화)을 사용함.
    """
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_vanilla_rnn_params():
    """vanilla RNN 파라미터를 초기화함.

    Vanilla RNN 업데이트 규칙:
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y

    가중치 행렬이 담긴 state_dict를 반환함.
    """
    params = {}
    params['W_xh'] = make_matrix(N_HIDDEN, VOCAB_SIZE)  # input-to-hidden
    params['W_hh'] = make_matrix(N_HIDDEN, N_HIDDEN)    # hidden-to-hidden (recurrent)
    params['b_h'] = [Value(0.0) for _ in range(N_HIDDEN)]  # hidden bias

    params['W_hy'] = make_matrix(VOCAB_SIZE, N_HIDDEN)  # hidden-to-output
    params['b_y'] = [Value(0.0) for _ in range(VOCAB_SIZE)]  # output bias

    return params


def init_gru_params():
    """GRU 파라미터를 초기화함.

    GRU 업데이트 규칙:
        z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})           # update gate
        r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1})           # reset gate
        h_candidate = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}))
        h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate       # 보간

    GRU는 gate 타입(z, r, h)마다 3개의 가중치 행렬을 써서 vanilla RNN 대비
    파라미터 수가 두 배지만, 차이를 만드는 건 파라미터 수가 아니라 gating 메커니즘임.
    """
    params = {}
    # Update gate
    params['W_xz'] = make_matrix(N_HIDDEN, VOCAB_SIZE)
    params['W_hz'] = make_matrix(N_HIDDEN, N_HIDDEN)

    # Reset gate
    params['W_xr'] = make_matrix(N_HIDDEN, VOCAB_SIZE)
    params['W_hr'] = make_matrix(N_HIDDEN, N_HIDDEN)

    # 후보 hidden state
    params['W_xh'] = make_matrix(N_HIDDEN, VOCAB_SIZE)
    params['W_hh'] = make_matrix(N_HIDDEN, N_HIDDEN)

    # Output projection (vanilla RNN과 같은 구조)
    params['W_hy'] = make_matrix(VOCAB_SIZE, N_HIDDEN)
    params['b_y'] = [Value(0.0) for _ in range(VOCAB_SIZE)]

    return params


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]], b: list[Value] = None) -> list[Value]:
    """행렬-벡터 곱: y = W @ x + b (bias는 선택적).

    가중치 행렬 W의 shape이 [n_out, n_in]이고 입력 벡터 x의 shape이 [n_in]일 때,
    shape [n_out]의 출력 y를 계산함.
    """
    y = [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]
    if b is not None:
        y = [y_i + b_i for y_i, b_i in zip(y, b)]
    return y


def softmax(logits: list[Value]) -> list[Value]:
    """수치적으로 안정적인 softmax: logit을 확률로 변환함.

    softmax는 이동 불변임: 임의의 c에 대해 softmax(x) = softmax(x - c).
    오버플로우 방지를 위해 exp() 전에 max(x)를 뺌.
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def safe_log(prob: Value) -> Value:
    """loss 계산에서 수치 안정성을 위한 클리핑된 로그.

    log(0)은 -inf를 반환하고 gradient 역전파를 깨뜨리므로 방지함.
    중요: gradient가 연산 그래프를 통해 흐르도록 `prob`을 자식 노드로 유지해야 함.
    """
    clamped = max(prob.data, 1e-10)
    # 그래프를 보존하면서 prob을 자식으로 갖는 log 노드를 수동으로 생성함.
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === VANILLA RNN FORWARD PASS ===

def vanilla_rnn_forward(
    x: list[Value], h_prev: list[Value], params: dict
) -> tuple[list[Value], list[Value]]:
    """단일 스텝 vanilla RNN forward pass.

    수식: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
          y_t = W_hy @ h_t + b_y

    recurrent 연결(W_hh @ h_{t-1})이 이걸 "recurrent"하게 만드는 것임 --
    hidden state가 이전 타임스텝의 정보를 전달함. 하지만 이 recurrence를 통한
    역전파에서 gradient가 W_hh에 반복 곱해지면서 W_hh의 spectral radius에 따라
    지수적 감소(vanishing gradient)나 폭발이 발생함.

    반환: (logits, new_hidden_state)
    """
    # 새로운 hidden state 계산
    h_input = linear(x, params['W_xh'])
    h_recurrent = linear(h_prev, params['W_hh'])
    h_combined = [h_i + h_r + params['b_h'][i] for i, (h_i, h_r) in enumerate(zip(h_input, h_recurrent))]
    h = [h_i.tanh() for h_i in h_combined]

    # 출력 logit 계산
    logits = linear(h, params['W_hy'], params['b_y'])

    return logits, h


# === GRU FORWARD PASS ===

def gru_forward(
    x: list[Value], h_prev: list[Value], params: dict
) -> tuple[list[Value], list[Value]]:
    """단일 스텝 GRU forward pass.

    수식:
        z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})           # update gate
        r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1})           # reset gate
        h_candidate = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}))
        h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate

    update gate z_t는 "gradient 고속도로" 역할을 함: z_t ≈ 0이면 h_t = h_{t-1}
    (이전 hidden state를 유지)이므로 dh_t/dh_{t-1} = 1임. 이 항등 gradient 흐름이
    vanishing gradient를 방지함 -- 도함수가 가중치 행렬에 곱해지지 않고 그냥
    통과함. 이것이 gating의 핵심 아이디어임.

    reset gate r_t는 후보 hidden state를 계산할 때 과거 정보를 얼마나 사용할지
    제어함. r_t ≈ 0이면 네트워크가 h_{t-1}을 무시하고 입력 x_t에서 새로 시작함.

    반환: (logits, new_hidden_state)
    """
    # Update gate: 새 state 대 이전 state의 비율을 제어함
    z_input = linear(x, params['W_xz'])
    z_recurrent = linear(h_prev, params['W_hz'])
    z = [(z_i + z_r).sigmoid() for z_i, z_r in zip(z_input, z_recurrent)]

    # Reset gate: 후보를 위해 이전 state를 얼마나 사용할지 제어함
    r_input = linear(x, params['W_xr'])
    r_recurrent = linear(h_prev, params['W_hr'])
    r = [(r_i + r_r).sigmoid() for r_i, r_r in zip(r_input, r_recurrent)]

    # 후보 hidden state: reset-gated된 이전 state로 계산됨
    # 원소별 곱셈(r_t * h_{t-1})이 현재 입력과 무관한 hidden state 성분을 "리셋"함.
    h_input = linear(x, params['W_xh'])
    h_reset = [r_i * h_i for r_i, h_i in zip(r, h_prev)]
    h_recurrent = linear(h_reset, params['W_hh'])
    h_candidate = [(h_i + h_r).tanh() for h_i, h_r in zip(h_input, h_recurrent)]

    # 이전 state와 후보를 보간함: z_t가 블렌딩을 제어함
    # z_t = 0이면: h_t = h_{t-1} (이전 state 유지, 새 입력을 "잊음")
    # z_t = 1이면: h_t = h_candidate (새 입력으로 완전히 업데이트)
    # 이 선형 보간이 gradient 고속도로를 만듦: dh_t/dh_{t-1}에
    # 가중치 행렬을 우회하는 (1 - z_t) 항이 포함됨.
    h = [(1 - z_i) * h_prev_i + z_i * h_cand_i
         for z_i, h_prev_i, h_cand_i in zip(z, h_prev, h_candidate)]

    # 출력 logit 계산
    logits = linear(h, params['W_hy'], params['b_y'])

    return logits, h


# === TRAINING FUNCTION ===

def train_rnn(
    docs: list[str],
    unique_chars: list[str],
    forward_fn,
    params: dict,
    model_name: str
) -> tuple[float, list[float]]:
    """RNN 모델(vanilla 또는 GRU)을 학습하고 gradient norm을 추적함.

    Args:
        docs: 학습 문서 (이름들)
        unique_chars: 어휘 (문자 리스트)
        forward_fn: vanilla_rnn_forward 또는 gru_forward
        params: 모델 파라미터 (state_dict)
        model_name: "Vanilla RNN" 또는 "GRU" (로깅용)

    Returns:
        (final_loss, gradient_norms_per_timestep)
    """
    BOS = len(unique_chars)
    VOCAB_SIZE_LOCAL = len(unique_chars) + 1

    # 모든 파라미터를 옵티마이저용 단일 리스트로 평탄화
    param_list = []
    for key, val in params.items():
        if isinstance(val, list) and isinstance(val[0], Value):
            param_list.extend(val)
        elif isinstance(val, list) and isinstance(val[0], list):
            for row in val:
                param_list.extend(row)

    print(f"Training {model_name}...")
    print(f"Parameters: {len(param_list):,}")

    final_loss_value = 0.0

    for step in range(NUM_STEPS):
        # 데이터셋을 순환하며 사용
        doc = docs[step % len(docs)]

        # 토큰화: [BOS, char_0, char_1, ..., char_n, BOS]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]

        # 시퀀스 길이로 자름
        seq_len = min(SEQ_LEN, len(tokens) - 1)

        # hidden state를 0으로 초기화
        h = [Value(0.0) for _ in range(N_HIDDEN)]

        # 시퀀스를 통한 forward pass
        losses = []
        for pos in range(seq_len):
            # 입력 토큰을 one-hot 인코딩
            x_onehot = [Value(1.0 if i == tokens[pos] else 0.0) for i in range(VOCAB_SIZE_LOCAL)]

            # 단일 타임스텝 forward
            logits, h = forward_fn(x_onehot, h, params)

            # loss 계산
            probs = softmax(logits)
            target = tokens[pos + 1]
            loss_t = -safe_log(probs[target])
            losses.append(loss_t)

        # 시퀀스에 대한 평균 loss
        loss = (1.0 / seq_len) * sum(losses)

        # Backward pass (Backpropagation Through Time - BPTT)
        loss.backward()

        # SGD 업데이트
        for param in param_list:
            param.data -= LEARNING_RATE * param.grad
            param.grad = 0.0

        final_loss_value = loss.data

        # 진행 상황 출력
        if (step + 1) % 200 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    print(f"{model_name} training complete. Final loss: {final_loss_value:.4f}\n")

    # === GRADIENT NORM TRACKING ===
    # vanishing gradient를 보여주기 위해 긴 시퀀스에서 gradient norm을 측정함.
    # 짧은 이름(~6글자)은 지수적 감쇠가 누적될 타임스텝이 충분하지 않아서
    # 극적인 gradient 감쇠를 보여주지 못함. 여러 이름을 연결해서
    # 더 명확한 시연을 위해 길이 SEQ_LEN의 시퀀스를 만듦.
    print(f"Measuring gradient norms for {model_name}...")

    # 이름들을 연결해서 SEQ_LEN에 도달할 때까지 긴 토큰 시퀀스를 만듦
    long_tokens = [BOS]
    for doc in docs:
        long_tokens.extend([unique_chars.index(ch) for ch in doc])
        long_tokens.append(BOS)
        if len(long_tokens) > SEQ_LEN:
            break
    seq_len = min(SEQ_LEN, len(long_tokens) - 1)

    # 긴 시퀀스를 통한 forward pass
    h = [Value(0.0) for _ in range(N_HIDDEN)]
    hidden_states = []

    for pos in range(seq_len):
        x_onehot = [Value(1.0 if i == long_tokens[pos] else 0.0) for i in range(VOCAB_SIZE_LOCAL)]
        logits, h = forward_fn(x_onehot, h, params)
        hidden_states.append(h)

    # 마지막 타임스텝에서만 loss를 계산함: gradient가 seq_len 타임스텝 전체를
    # 통과해야 함. vanishing gradient가 가장 심한 시나리오 -- loss 신호가
    # 전체 시퀀스를 횡단해야 함.
    probs = softmax(logits)
    target = long_tokens[seq_len]
    loss = -safe_log(probs[target])

    # Backward pass
    loss.backward()

    # 각 hidden state의 gradient L2 norm을 계산함
    # ||dL/dh_t|| = sqrt(sum_i (dL/dh_t[i])^2)
    # vanilla RNN: t가 감소할수록 (loss에서 멀어질수록) 지수적 감쇠가 예상됨
    # GRU: gate를 통한 gradient 고속도로 덕분에 더 균일한 norm이 예상됨
    gradient_norms = []
    for h_t in hidden_states:
        norm_sq = sum(h_i.grad ** 2 for h_i in h_t)
        norm = math.sqrt(norm_sq)
        gradient_norms.append(norm)

    # gradient norm 출력
    print(f"Gradient norms per timestep (sequence length {seq_len}):")
    for t, norm in enumerate(gradient_norms):
        bar = "#" * min(50, int(norm * 100))
        print(f"  t={t:>2}: ||dL/dh_t|| = {norm:.6f}  {bar}")

    # 비율 계산: first / last (gradient가 역방향으로 얼마나 감쇠하는지 측정)
    # < 1이면 gradient가 시간을 거슬러 역방향으로 가면서 소멸됨 (first < last)
    # 이 비율이 0에 가까울수록 vanishing gradient 문제가 심각함.
    if gradient_norms[-1] > 1e-10:
        ratio = gradient_norms[0] / gradient_norms[-1]
    else:
        ratio = 0.0
    print(f"Gradient norm ratio (first/last): {ratio:.6f}")
    print(f"  (< 0.01 = severe vanishing, > 0.1 = gradient highway active)\n")

    return final_loss_value, gradient_norms


# === INFERENCE FUNCTION ===

def generate_names(
    params: dict,
    forward_fn,
    unique_chars: list[str],
    num_samples: int = 10,
    model_name: str = "Model"
) -> list[str]:
    """학습된 RNN 모델에서 이름을 생성함."""
    BOS = len(unique_chars)
    VOCAB_SIZE_LOCAL = len(unique_chars) + 1

    print(f"Generating {num_samples} samples from {model_name}:")

    samples = []
    for _ in range(num_samples):
        h = [Value(0.0) for _ in range(N_HIDDEN)]
        token_id = BOS
        generated = []

        for pos in range(SEQ_LEN):
            # 현재 토큰을 one-hot 인코딩
            x_onehot = [Value(1.0 if i == token_id else 0.0) for i in range(VOCAB_SIZE_LOCAL)]

            # Forward
            logits, h = forward_fn(x_onehot, h, params)

            # 확률에서 샘플링
            probs = softmax(logits)
            token_id = random.choices(
                range(VOCAB_SIZE_LOCAL),
                weights=[p.data for p in probs]
            )[0]

            # BOS(시퀀스 끝)이면 중단
            if token_id == BOS:
                break

            generated.append(unique_chars[token_id])

        name = ''.join(generated)
        samples.append(name)
        print(f"  {''.join(generated)}")

    print()
    return samples


# === MAIN ===

if __name__ == "__main__":
    # -- 데이터 로드 및 준비 --
    print("Loading data...")
    all_docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(all_docs)

    # 작은 학습 서브셋을 사용해서 각 이름이 500 스텝 동안 여러 번 보이게 함.
    # 200개 이름과 500 스텝이면 각 이름이 ~2.5번 보임 — 수천 번의 gradient
    # 스텝 없이도 character-level 패턴을 학습하기에 충분함.
    docs = all_docs[:TRAIN_SIZE]

    # 모든 이름에서 어휘를 구축함 (문자를 누락하지 않기 위해)
    unique_chars = sorted(set(''.join(all_docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(all_docs)} documents, training on {len(docs)}")
    print(f"Vocabulary size: {VOCAB_SIZE} (characters + BOS token)\n")

    # === TRAIN VANILLA RNN ===
    vanilla_params = init_vanilla_rnn_params()
    vanilla_loss, vanilla_grad_norms = train_rnn(
        docs, unique_chars, vanilla_rnn_forward, vanilla_params, "Vanilla RNN"
    )

    # === TRAIN GRU ===
    gru_params = init_gru_params()
    gru_loss, gru_grad_norms = train_rnn(
        docs, unique_chars, gru_forward, gru_params, "GRU"
    )

    # === COMPARISON TABLE ===
    print("=" * 70)
    print("COMPARISON: Vanilla RNN vs GRU")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Vanilla RNN':<15} | {'GRU':<15}")
    print("-" * 70)
    print(f"{'Final Loss':<30} | {vanilla_loss:<15.4f} | {gru_loss:<15.4f}")

    # Gradient norm 비율: first/last로 역방향 gradient 감쇠를 측정함
    # 낮은 비율 = vanishing gradient가 더 심함 (gradient가 역방향으로 더 많이 감쇠)
    vanilla_ratio = vanilla_grad_norms[0] / vanilla_grad_norms[-1] if vanilla_grad_norms[-1] > 1e-10 else 0.0
    gru_ratio = gru_grad_norms[0] / gru_grad_norms[-1] if gru_grad_norms[-1] > 1e-10 else 0.0

    print(f"{'Gradient Norm Ratio':<30} | {vanilla_ratio:<15.6f} | {gru_ratio:<15.6f}")
    print(f"{'(first/last, higher=better)':<30} |                 |                ")
    print("-" * 70)

    # 차이가 중요한 이유
    print("\nWhy the gradient norm ratio matters:")
    print("  Vanilla RNN: Gradient norms decay exponentially due to repeated")
    print("               multiplication by W_hh. Spectral radius < 1 causes")
    print("               gradients to vanish as they propagate backward through time.")
    print("  GRU:         Update gate creates 'gradient highways' where dh_t/dh_{t-1} ≈ 1")
    print("               when z_t ≈ 0. This identity connection bypasses weight matrices,")
    print("               preserving gradient magnitude across long sequences.\n")

    # === INFERENCE ===
    print("=" * 70)
    print("GENERATED SAMPLES")
    print("=" * 70)
    print()

    vanilla_samples = generate_names(
        vanilla_params, vanilla_rnn_forward, unique_chars, num_samples=10, model_name="Vanilla RNN"
    )

    gru_samples = generate_names(
        gru_params, gru_forward, unique_chars, num_samples=10, model_name="GRU"
    )

    # === HISTORICAL CONTEXT ===
    print("=" * 70)
    print("HISTORICAL ARC")
    print("=" * 70)
    print("  1990s:  Vanilla RNNs introduced — theoretically powerful, but gradients")
    print("          vanish in practice, limiting them to short sequences (~10 steps).")
    print("  1997:   LSTM (Long Short-Term Memory) introduced gating to solve the")
    print("          vanishing gradient problem. Became the standard for sequence modeling.")
    print("  2014:   GRU (Gated Recurrent Unit) simplified LSTM's 3 gates to 2, achieving")
    print("          similar performance with fewer parameters and faster training.")
    print("  2017:   Transformers (Attention Is All You Need) replaced recurrence entirely,")
    print("          using attention for O(1) path length between any two positions.")
    print("  Today:  RNNs are largely historical, but the gating principle (learned routing")
    print("          of gradients) lives on in modern architectures like state-space models.")
