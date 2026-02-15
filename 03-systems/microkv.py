"""
autoregressive 생성이 매 스텝마다 중복 연산을 왜 하는지, 그리고 KV cache가
시퀀스에 걸쳐 key/value 프로젝션을 메모이제이션해서 그 중복을 어떻게 제거하는지 보여줌.
"""
# Reference: Pope et al., "Efficiently Scaling Transformer Inference" (2022) for KV cache
# analysis. Kwon et al., "Efficient Memory Management for Large Language Model Serving
# with PagedAttention" (2023) for paged allocation. Architecture follows the microgpt
# pattern (Radford et al., 2019) with pedagogical simplifications.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# 모델 아키텍처 — cache 메커니즘 시연을 위한 최소한의 트랜스포머.
# 모델은 랜덤이 아닌 출력만 생성하면 됨; 생성 품질은 부차적임.
N_EMBD = 16
N_HEAD = 2
N_LAYER = 1
BLOCK_SIZE = 32
HEAD_DIM = N_EMBD // N_HEAD  # 8

# 학습
LEARNING_RATE = 0.01
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
NUM_STEPS = 300

# 추론 비교
GEN_LEN = 16  # 비교를 위해 생성할 문자 수
PAGE_BLOCK_SIZE = 4  # paged attention 시뮬레이션에서 블록당 위치 수

# 데이터
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 참고: 프로덕션 KV cache는 수십 개 레이어에 걸쳐 수천 개 위치를 128차원 헤드로 저장함.
# 우리의 토이 차원(1 레이어, 8차원 헤드)은 알고리즘 구조를 보존하면서
# 실행 시간을 1분 이내로 유지함.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """학습 코퍼스를 다운로드하고 파싱함."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


# === SCALAR AUTOGRAD ENGINE (compact) ===
# 학습에만 사용됨. 학습 완료 후 가중치를 일반 float로 추출하고
# 순수 산술로 추론을 실행함 -- KV cache 비교를 autograd 오버헤드에서
# 분리하고 곱셈 횟수를 정확하게 셈.

class Value:
    """역전파 자동 미분이 있는 스칼라. docs/autograd-interface.md 스펙 참조."""
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

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self) -> Value:
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self) -> None:
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
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# 소형 Value 클래스로 작은 모델 학습에만 사용됨. 추론 비교는
# 추출된 일반 float로 동작해서 autograd 오버헤드 없이 곱셈 횟수를 정확히 셈.
# 정규 인터페이스는 docs/autograd-interface.md 참조.


# === TRAINING HELPERS (Value-based) ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]

def linear_v(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]

def softmax_v(logits: list[Value]) -> list[Value]:
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm_v(x: list[Value]) -> list[Value]:
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def safe_log(prob: Value) -> Value:
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def gpt_forward_train(
    token_id: int, pos_id: int,
    keys: list[list[list[Value]]], values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
) -> list[Value]:
    """학습용 단일 토큰 순전파 (Value 기반)."""
    x = [t + p for t, p in zip(params['wte'][token_id], params['wpe'][pos_id])]
    x = rmsnorm_v(x)
    for li in range(N_LAYER):
        x_res = x
        x = rmsnorm_v(x)
        q = linear_v(x, params[f'l{li}.wq'])
        k = linear_v(x, params[f'l{li}.wk'])
        v = linear_v(x, params[f'l{li}.wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn: list[Value] = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = q[hs:hs + HEAD_DIM]
            k_h = [kt[hs:hs + HEAD_DIM] for kt in keys[li]]
            v_h = [vt[hs:hs + HEAD_DIM] for vt in values[li]]
            scores = [sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                      for t in range(len(k_h))]
            weights = softmax_v(scores)
            x_attn.extend([sum(weights[t] * v_h[t][j] for t in range(len(v_h)))
                           for j in range(HEAD_DIM)])
        x = linear_v(x_attn, params[f'l{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x
        x = rmsnorm_v(x)
        x = linear_v(x, params[f'l{li}.fc1'])
        x = [xi.relu() for xi in x]
        x = linear_v(x, params[f'l{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear_v(x, params['lm_head'])


# === PLAIN-FLOAT INFERENCE HELPERS ===
# 학습 후 Value.data를 일반 float 중첩 리스트로 추출함.
# 아래의 모든 추론 비교 함수는 float로 동작하며 곱셈 횟수를 셈.

def extract(w: list[list[Value]]) -> list[list[float]]:
    """autograd 래퍼를 제거함: list[list[Value]] -> list[list[float]]."""
    return [[v.data for v in row] for row in w]


def linear_f(x: list[float], w: list[list[float]], counter: list[int]) -> list[float]:
    """일반 float 행렬-벡터 곱셈. 모든 스칼라 곱셈을 카운트함."""
    counter[0] += len(w) * len(x)
    return [sum(w[i][j] * x[j] for j in range(len(x))) for i in range(len(w))]


def softmax_f(logits: list[float]) -> list[float]:
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def rmsnorm_f(x: list[float]) -> list[float]:
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# === INFERENCE WITHOUT KV CACHE ===
# 각 생성 스텝마다 모든 위치의 Q/K/V 프로젝션을 처음부터 다시 계산함.
# 매 스텝을 독립적으로 처리하는 것처럼 attention이 동작하는 방식임:
# 전체 시퀀스를 입력하고, 전부에 대해 attend하고, 중간 상태를 버리고, 반복함.
# T 스텝의 총 연산량: sum(t * C_proj + t^2 * C_attn for t in 1..T) ~ O(T^3)

def generate_no_cache(
    prompt_tok: int, wf: dict[str, list[list[float]]],
    vocab_size: int, gen_len: int,
) -> tuple[list[int], list[int]]:
    """KV cache 없이 토큰을 생성함. (tokens, muls_per_step)을 반환함."""
    tokens = [prompt_tok]
    muls_per_step: list[int] = []

    for step in range(gen_len):
        counter = [0]
        seq = tokens  # 매 스텝마다 전체 시퀀스를 처음부터 다시 처리함
        seq_len = len(seq)

        # 모든 위치를 임베딩함
        embeddings: list[list[float]] = []
        for pos, tid in enumerate(seq):
            x = [wf['wte'][tid][j] + wf['wpe'][pos][j] for j in range(N_EMBD)]
            embeddings.append(rmsnorm_f(x))

        # 트랜스포머 레이어 — 모든 위치의 Q, K, V를 다시 계산함
        hiddens = [row[:] for row in embeddings]
        for li in range(N_LAYER):
            residuals = [row[:] for row in hiddens]
            normed = [rmsnorm_f(h) for h in hiddens]

            # 모든 위치를 Q, K, V로 프로젝션함 — 이게 중복 연산임.
            # 위치 0..(t-1)은 이전 스텝에서 이미 프로젝션했었음.
            all_q = [linear_f(normed[p], wf[f'l{li}.wq'], counter) for p in range(seq_len)]
            all_k = [linear_f(normed[p], wf[f'l{li}.wk'], counter) for p in range(seq_len)]
            all_v = [linear_f(normed[p], wf[f'l{li}.wv'], counter) for p in range(seq_len)]

            # 전체 시퀀스에 대한 causal multi-head attention
            attn_out: list[list[float]] = []
            for pos in range(seq_len):
                head_cat: list[float] = []
                for h in range(N_HEAD):
                    hs = h * HEAD_DIM
                    q_h = all_q[pos][hs:hs + HEAD_DIM]
                    # Causal: 위치 0..pos에만 attend
                    scores: list[float] = []
                    for t in range(pos + 1):
                        k_h = all_k[t][hs:hs + HEAD_DIM]
                        dot = sum(q_h[j] * k_h[j] for j in range(HEAD_DIM))
                        counter[0] += HEAD_DIM
                        scores.append(dot / (HEAD_DIM ** 0.5))
                    weights = softmax_f(scores)
                    for j in range(HEAD_DIM):
                        val = 0.0
                        for t in range(pos + 1):
                            val += weights[t] * all_v[t][hs + j]
                            counter[0] += 1
                        head_cat.append(val)
                attn_out.append(head_cat)

            # 출력 프로젝션 + 잔차
            for pos in range(seq_len):
                projected = linear_f(attn_out[pos], wf[f'l{li}.wo'], counter)
                hiddens[pos] = [a + b for a, b in zip(projected, residuals[pos])]

            # MLP + 잔차
            for pos in range(seq_len):
                res2 = hiddens[pos][:]
                h = rmsnorm_f(hiddens[pos])
                h = linear_f(h, wf[f'l{li}.fc1'], counter)
                h = [max(0.0, v) for v in h]
                h = linear_f(h, wf[f'l{li}.fc2'], counter)
                hiddens[pos] = [a + b for a, b in zip(h, res2)]

        # 마지막 위치에서만 logits 계산
        logits = linear_f(hiddens[-1], wf['lm_head'], counter)
        probs = softmax_f(logits)
        next_tok = max(range(vocab_size), key=lambda i: probs[i])
        tokens.append(next_tok)
        muls_per_step.append(counter[0])

    return tokens[1:], muls_per_step


# === INFERENCE WITH KV CACHE ===
# 각 스텝에서 새 토큰에 대해서만 Q/K/V를 계산함. K와 V를 cache에 추가함.
# Attention: Q_new가 캐시된 모든 K (0..t), V (0..t)에 attend함.
# 스텝당 연산량: C_proj + t * C_attn ~ O(t). T 스텝 총합: O(T^2) — 한 차수 향상.
# 핵심 통찰: autoregressive 디코딩에서 과거 토큰의 K, V 프로젝션은 절대 변하지 않음.
# 이걸 다시 계산하는 건 순수한 낭비임 — KV cache는 선형 프로젝션의 메모이제이션임.

def generate_with_cache(
    prompt_tok: int, wf: dict[str, list[list[float]]],
    vocab_size: int, gen_len: int,
) -> tuple[list[int], list[int], list[int]]:
    """KV cache로 토큰을 생성함. (tokens, muls_per_step, cache_sizes)를 반환함."""
    tokens: list[int] = []
    muls_per_step: list[int] = []
    cache_sizes: list[int] = []

    # KV cache: 각 레이어와 위치에 대한 프로젝션된 K, V 벡터를 저장함.
    # Shape: kv_cache[layer] = {'k': 벡터 리스트, 'v': 벡터 리스트}
    kv_cache: list[dict[str, list[list[float]]]] = [
        {'k': [], 'v': []} for _ in range(N_LAYER)
    ]

    current_tok = prompt_tok
    for step in range(gen_len):
        counter = [0]
        pos = step

        # 새 토큰만 임베딩함 — 이전 임베딩은 재계산 불필요
        x = [wf['wte'][current_tok][j] + wf['wpe'][pos][j] for j in range(N_EMBD)]
        x = rmsnorm_f(x)

        for li in range(N_LAYER):
            x_res = x[:]
            x = rmsnorm_f(x)

            # 새 토큰만 프로젝션함 — cache가 연산을 절약하는 부분임.
            # cache 없이: 모든 t개 토큰을 프로젝션. cache 있으면: 1개 토큰만 프로젝션.
            q = linear_f(x, wf[f'l{li}.wq'], counter)
            k = linear_f(x, wf[f'l{li}.wk'], counter)
            v = linear_f(x, wf[f'l{li}.wv'], counter)

            # 새 K, V를 cache에 추가 (cache가 스텝마다 하나씩 커짐)
            kv_cache[li]['k'].append(k)
            kv_cache[li]['v'].append(v)

            # Attention: 새 토큰의 Q가 캐시된 모든 K/V에 attend함
            head_cat: list[float] = []
            cached_len = len(kv_cache[li]['k'])
            for h in range(N_HEAD):
                hs = h * HEAD_DIM
                q_h = q[hs:hs + HEAD_DIM]
                scores: list[float] = []
                for t in range(cached_len):
                    k_h = kv_cache[li]['k'][t][hs:hs + HEAD_DIM]
                    dot = sum(q_h[j] * k_h[j] for j in range(HEAD_DIM))
                    counter[0] += HEAD_DIM
                    scores.append(dot / (HEAD_DIM ** 0.5))
                weights = softmax_f(scores)
                for j in range(HEAD_DIM):
                    val = 0.0
                    for t in range(cached_len):
                        val += weights[t] * kv_cache[li]['v'][t][hs + j]
                        counter[0] += 1
                    head_cat.append(val)

            x = linear_f(head_cat, wf[f'l{li}.wo'], counter)
            x = [a + b for a, b in zip(x, x_res)]
            x_res = x[:]
            x = rmsnorm_f(x)
            x = linear_f(x, wf[f'l{li}.fc1'], counter)
            x = [max(0.0, v) for v in x]
            x = linear_f(x, wf[f'l{li}.fc2'], counter)
            x = [a + b for a, b in zip(x, x_res)]

        logits = linear_f(x, wf['lm_head'], counter)
        probs = softmax_f(logits)
        next_tok = max(range(vocab_size), key=lambda i: probs[i])
        tokens.append(next_tok)
        current_tok = next_tok
        muls_per_step.append(counter[0])

        # Cache 메모리: 캐시된 위치당 2 (K+V) * n_layer * n_embd floats
        total_cached_floats = 2 * N_LAYER * N_EMBD * len(kv_cache[0]['k'])
        cache_sizes.append(total_cached_floats)

    return tokens, muls_per_step, cache_sizes


# === PAGED ATTENTION SIMULATION ===
# 프로덕션 시스템(vLLM)은 시퀀스 길이를 미리 알 수 없고 가변적이라
# 모든 시퀀스의 KV cache에 연속 메모리를 미리 할당할 수 없음.
# Paged attention은 OS 가상 메모리 아이디어를 차용함: 고정 크기 블록을
# 필요에 따라 할당하고, 논리적 위치를 페이지 테이블을 통해 물리적 블록에 매핑함.
# 이렇게 하면 과다 할당으로 인한 단편화를 제거하고, 시퀀스 간 물리 블록
# 공유가 가능해짐 (예: 공유 프리픽스).

def simulate_paged_attention(seq_len: int, block_size: int) -> None:
    """paged attention이 cache 블록을 어떻게 할당하고 매핑하는지 시연함."""
    print(f"\n=== Paged Attention Simulation ===")
    print(f"Block size: {block_size} positions | Sequence length: {seq_len}")
    print(f"Each block holds {block_size} positions of KV data\n")

    # 페이지 테이블: 논리적 블록 인덱스 -> 물리적 블록 인덱스 매핑
    # 물리적 블록은 필요에 따라 풀에서 할당됨
    page_table: list[int] = []
    next_physical_block = 0

    print("Allocation trace:")
    for pos in range(seq_len):
        logical_block = pos // block_size
        slot_in_block = pos % block_size

        # 새 논리적 블록에 진입할 때 새 물리적 블록을 할당
        if logical_block >= len(page_table):
            page_table.append(next_physical_block)
            print(f"  Position {pos:>2}: new block needed -> "
                  f"logical block {logical_block} -> physical block {next_physical_block}")
            next_physical_block += 1
        else:
            print(f"  Position {pos:>2}: slot {slot_in_block} in "
                  f"logical block {logical_block} (physical {page_table[logical_block]})")

    print(f"\nPage table (logical -> physical):")
    for i, phys in enumerate(page_table):
        start = i * block_size
        end = min(start + block_size - 1, seq_len - 1)
        status = "FULL" if (i + 1) * block_size <= seq_len else f"{seq_len - i * block_size}/{block_size}"
        print(f"  Logical block {i} -> Physical block {phys} "
              f"[positions {start}-{end}] {status}")

    # 참고: 프로덕션에서는 물리적 블록이 시퀀스 간에 공유됨. 같은 시스템 메시지로
    # 시작하는 두 프롬프트가 공유 프리픽스에 대해 같은 물리적 블록을 재사용함
    # — vLLM의 copy-on-write가 KV 데이터 복제를 피함. 여기선 단일 시퀀스
    # 할당만 시뮬레이션함; 멀티 시퀀스 공유가 대규모에서의 실제 메모리 절약임.
    wasted = len(page_table) * block_size - seq_len
    print(f"\nBlocks allocated: {len(page_table)} ({len(page_table) * block_size} slots)")
    print(f"Slots used: {seq_len} | Wasted: {wasted} "
          f"({100 * wasted / (len(page_table) * block_size):.0f}% internal fragmentation)")


# === MAIN ===

if __name__ == "__main__":
    # -- 데이터 로드 및 어휘 구축 --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1
    print(f"Loaded {len(docs)} documents, vocab size: {VOCAB_SIZE}")

    # -- 소형 모델 학습 --
    # 학습이 핵심이 아님 — cache 유무 비교가 의미 있으려면
    # 결정론적이고 랜덤이 아닌 출력을 만드는 모델이 필요할 뿐임.
    print(f"\nTraining tiny model ({NUM_STEPS} steps)...")
    params: dict[str, list[list[Value]]] = {}
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)
    for li in range(N_LAYER):
        params[f'l{li}.wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'l{li}.wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'l{li}.wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'l{li}.wo'] = make_matrix(N_EMBD, N_EMBD)
        params[f'l{li}.fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'l{li}.fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    param_list = [p for w in params.values() for row in w for p in row]
    m_state = [0.0] * len(param_list)
    v_state = [0.0] * len(param_list)

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)
        keys_t = [[] for _ in range(N_LAYER)]
        vals_t = [[] for _ in range(N_LAYER)]
        losses: list[Value] = []
        for pos in range(seq_len):
            logits = gpt_forward_train(tokens[pos], pos, keys_t, vals_t, params)
            probs = softmax_v(logits)
            losses.append(-safe_log(probs[tokens[pos + 1]]))
        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, p in enumerate(param_list):
            m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * p.grad
            v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    print(f"Training complete. Final loss: {loss.data:.4f}")

    # -- 가중치를 일반 float로 추출 --
    wf: dict[str, list[list[float]]] = {k: extract(v) for k, v in params.items()}

    # -- 동일한 프롬프트에 대해 두 추론 방식 실행 --
    prompt_tok = BOS
    print(f"\n=== KV-Cache Comparison ===")
    print(f"Generating {GEN_LEN}-character sequence from BOS token\n")

    t0 = time.time()
    toks_no_cache, muls_no = generate_no_cache(prompt_tok, wf, VOCAB_SIZE, GEN_LEN)
    time_no = time.time() - t0

    t0 = time.time()
    toks_cached, muls_yes, cache_sizes = generate_with_cache(prompt_tok, wf, VOCAB_SIZE, GEN_LEN)
    time_cached = time.time() - t0

    # -- 동일한 출력 검증 --
    # 두 방식은 수학적으로 동일한 함수를 계산함. KV cache는
    # 근사가 아닌 연산 단축임 — 출력이 정확히 일치해야 함.
    assert toks_no_cache == toks_cached, (
        f"Output mismatch: no-cache produced {toks_no_cache}, cache produced {toks_cached}"
    )

    # -- 스텝별 비교 출력 --
    name_no = [unique_chars[t] if t != BOS else '.' for t in toks_no_cache]
    name_yes = [unique_chars[t] if t != BOS else '.' for t in toks_cached]

    header = f"{'Step':>4}  {'No Cache (muls)':>16}  {'With Cache (muls)':>18}  {'Speedup':>8}  {'Match':>5}"
    print(header)
    print("-" * len(header))
    for i in range(GEN_LEN):
        ratio = muls_no[i] / muls_yes[i] if muls_yes[i] > 0 else 0
        match = "yes" if toks_no_cache[i] == toks_cached[i] else "NO"
        print(f"{i + 1:>4}  {muls_no[i]:>16,}  {muls_yes[i]:>18,}  {ratio:>7.1f}x  {match:>5}")

    total_no = sum(muls_no)
    total_yes = sum(muls_yes)
    overall_ratio = total_no / total_yes if total_yes > 0 else 0
    print(f"\nTotal multiplies -- No cache: {total_no:,} | With cache: {total_yes:,} | "
          f"Ratio: {overall_ratio:.1f}x")
    print(f"Wall time -- No cache: {time_no:.3f}s | With cache: {time_cached:.3f}s")

    generated_str = ''.join(name_yes)
    print(f"\nGenerated: \"{generated_str}\" (both methods identical)")

    # -- 메모리 증가 분석 --
    # KV cache는 위치당 레이어당 2개의 벡터(K와 V)를 저장하며, 각각 n_embd 크기임.
    # 메모리 증가는 시퀀스 길이에 대해 엄격하게 선형임 — 이차적 폭증 없음.
    # 이 선형 증가가 긴 컨텍스트 모델(100K+ 토큰)이 메모리 바운드인 이유임:
    # d_model=4096, 40 레이어, 100K 토큰이면 cache는 float16으로 ~32GB임.
    floats_per_pos = 2 * N_LAYER * N_EMBD
    print(f"\n=== Memory Growth ===")
    print(f"{'Position':>8}   {'Cache Size (floats)':>20}   {'Cache Size (bytes, float32)':>28}")
    print("-" * 62)
    for i in range(GEN_LEN):
        n_floats = cache_sizes[i]
        n_bytes = n_floats * 4
        print(f"{i + 1:>8}   {n_floats:>20,}   {n_bytes:>28,}")

    print(f"\nGrowth: linear O(n) -- {floats_per_pos} floats per position "
          f"(2 * {N_LAYER} layer * {N_EMBD} embd)")
    print(f"Signpost: LLaMA-2 70B with 80 layers, 8192 embd, 4K context = "
          f"~5.2 GB KV cache in float16.")
    print(f"This is why KV cache memory, not compute, is the bottleneck for long sequences.")

    # -- Paged attention --
    simulate_paged_attention(GEN_LEN, PAGE_BLOCK_SIZE)
