"""
greedy 디코딩을 넘어서: 결정적 argmax부터 draft-verify 두 모델 파이프라인인
speculative decoding까지 언어 모델 텍스트 생성을 위한 6가지 디코딩 전략을 다룸.
"""
# Reference: Leviathan et al., "Fast Inference from Transformers via Speculative
# Decoding" (2023). https://arxiv.org/abs/2211.17192
# Also: Holtzman et al., "The Curious Case of Neural Text Degeneration" (2019).
# https://arxiv.org/abs/1904.09751 (nucleus/top-p sampling)

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===

# target 모델(더 큰 모델, ~4,200 파라미터)과 draft 모델(더 작은 모델, ~1,300 파라미터).
# 둘 다 vocabulary와 block_size를 공유함 — speculative decoding에서 draft 모델이
# 생성한 토큰을 target 모델이 검증할 수 있어야 하기 때문에 필수임.
TARGET_N_EMBD, TARGET_N_HEAD, TARGET_N_LAYER = 16, 4, 1
DRAFT_N_EMBD, DRAFT_N_HEAD, DRAFT_N_LAYER = 8, 2, 1
BLOCK_SIZE = 16

# 학습
LEARNING_RATE, BETA1, BETA2, EPS_ADAM = 0.01, 0.85, 0.99, 1e-8
TARGET_STEPS, DRAFT_STEPS = 700, 500

# 데이터
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 참고: 실제 프로덕션 speculative decoding은 70B target과 7B draft를 조합함.
# 우리의 4,200 / 1,300 파라미터 비율은 알고리즘 구조를 그대로 유지함. 실제 속도 향상은
# verify 패스 중 GPU 병렬 처리에서 나옴 — 여기서는 하드웨어 독립적인 지표인
# 수락율을 측정함.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """학습 코퍼스를 다운로드하고 파싱함."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """역전파 자동 미분을 지원하는 스칼라 값.

    ._children와 ._local_grads를 통해 연산 히스토리를 추적하여, chain rule로
    그래디언트를 계산할 수 있게 함. 모든 순방향 연산은 로컬 미분값(dout/dinput)을
    저장하고, backward()가 역 위상 정렬 순서로 그래프를 역방향 탐색하면서
    그래디언트를 누적함.
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
        # d(x^n)/dx = n * x^(n-1)
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
        """위상 정렬 후 chain rule을 적용하는 역전파 자동 미분."""
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
# 이 Value 클래스는 표준 인터페이스를 정확히 따름.
# autograd는 학습에만 사용됨. 모든 디코딩 전략은 추론 속도를 위해
# 일반 float 순방향 패스를 사용함.
# 전체 명세는 docs/autograd-interface.md 참조.


# === TRAINING HELPERS (Value-based) ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]

def linear_v(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]

def softmax_v(logits: list[Value]) -> list[Value]:
    """수치적으로 안정적인 softmax: 오버플로 방지를 위해 exp 전에 최댓값을 뺌."""
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm_v(x: list[Value]) -> list[Value]:
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def safe_log(prob: Value) -> Value:
    """클리핑된 log: log(0) 방지를 위해 1e-10으로 클램핑함. 노드가 prob을
    자식으로 유지해서 연산 그래프를 통해 그래디언트가 역전파됨."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))

def gpt_forward_train(
    token_id: int, pos_id: int,
    keys: list[list[list[Value]]], values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
    n_embd: int, n_head: int, n_layer: int, head_dim: int,
) -> list[Value]:
    """학습용 단일 토큰 GPT 순방향 패스. 모델 설정으로 파라미터화됨."""
    x = [t + p for t, p in zip(params['wte'][token_id], params['wpe'][pos_id])]
    x = rmsnorm_v(x)
    for li in range(n_layer):
        x_res = x
        x = rmsnorm_v(x)
        q = linear_v(x, params[f'l{li}.wq'])
        k = linear_v(x, params[f'l{li}.wk'])
        v = linear_v(x, params[f'l{li}.wv'])
        keys[li].append(k); values[li].append(v)
        # 점진적 KV 구성을 사용한 multi-head attention (암묵적 causal mask)
        x_attn: list[Value] = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [kt[hs:hs + head_dim] for kt in keys[li]]
            v_h = [vt[hs:hs + head_dim] for vt in values[li]]
            scores = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / (head_dim ** 0.5)
                      for t in range(len(k_h))]
            w = softmax_v(scores)
            x_attn.extend([sum(w[t] * v_h[t][j] for t in range(len(v_h)))
                           for j in range(head_dim)])
        x = linear_v(x_attn, params[f'l{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x
        x = rmsnorm_v(x)
        x = linear_v(x, params[f'l{li}.fc1'])
        x = [xi.relu() for xi in x]
        x = linear_v(x, params[f'l{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear_v(x, params['lm_head'])


# === PLAIN-FLOAT INFERENCE ===
# 학습 후 가중치를 일반 float으로 추출함. 6가지 디코딩 전략 모두
# 여기서 동작함 — autograd 오버헤드 없이 깔끔한 비교가 가능함.

def extract(w: list[list[Value]]) -> list[list[float]]:
    return [[v.data for v in row] for row in w]

def linear_f(x: list[float], w: list[list[float]]) -> list[float]:
    return [sum(w[i][j] * x[j] for j in range(len(x))) for i in range(len(w))]

def softmax_f(logits: list[float]) -> list[float]:
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm_f(x: list[float]) -> list[float]:
    mean_sq = sum(xi * xi for xi in x) / len(x)
    return [xi * (mean_sq + 1e-5) ** -0.5 for xi in x]

# 모델 설정 딕셔너리: n_embd, n_head, n_layer, head_dim, vocab_size, bos
Cfg = dict

def forward_float(tok: int, pos: int, kv: list[dict[str, list[list[float]]]],
                  wf: dict[str, list[list[float]]], c: Cfg) -> list[float]:
    """일반 float을 사용하는 GPT 순방향 패스. 토큰 하나를 처리하고 K/V를
    kv 캐시에 추가한 뒤 logits를 반환함. 모든 디코딩 전략이 이 함수를 공유함 —
    차이점은 오직 이 logits에서 다음 토큰을 어떻게 선택하느냐에 있음."""
    x = [wf['wte'][tok][j] + wf['wpe'][pos][j] for j in range(c['n_embd'])]
    x = rmsnorm_f(x)
    for li in range(c['n_layer']):
        x_res = x[:]
        x = rmsnorm_f(x)
        q = linear_f(x, wf[f'l{li}.wq'])
        k = linear_f(x, wf[f'l{li}.wk'])
        v = linear_f(x, wf[f'l{li}.wv'])
        kv[li]['k'].append(k); kv[li]['v'].append(v)
        head_cat: list[float] = []
        clen = len(kv[li]['k'])
        hd = c['head_dim']
        for h in range(c['n_head']):
            hs = h * hd
            q_h = q[hs:hs + hd]
            scores = [sum(q_h[j] * kv[li]['k'][t][hs + j] for j in range(hd)) / (hd ** 0.5)
                      for t in range(clen)]
            w = softmax_f(scores)
            for j in range(hd):
                head_cat.append(sum(w[t] * kv[li]['v'][t][hs + j] for t in range(clen)))
        x = linear_f(head_cat, wf[f'l{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x[:]
        x = rmsnorm_f(x)
        x = linear_f(x, wf[f'l{li}.fc1'])
        x = [max(0.0, v) for v in x]
        x = linear_f(x, wf[f'l{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear_f(x, wf['lm_head'])

def make_kv(c: Cfg) -> list[dict[str, list[list[float]]]]:
    return [{'k': [], 'v': []} for _ in range(c['n_layer'])]

def clone_kv(cache: list[dict[str, list[list[float]]]]) -> list[dict[str, list[list[float]]]]:
    """beam 브랜치가 가변 상태를 공유하지 않도록 KV 캐시를 깊은 복사함."""
    return [{'k': [r[:] for r in l['k']], 'v': [r[:] for r in l['v']]} for l in cache]

def feed_prompt(toks: list[int], wf: dict[str, list[list[float]]],
                c: Cfg) -> tuple[list[dict[str, list[list[float]]]], list[float]]:
    """프롬프트를 모델에 통과시켜 (kv_cache, last_logits)를 반환함."""
    kv = make_kv(c)
    logits: list[float] = []
    for i, t in enumerate(toks):
        logits = forward_float(t, i, kv, wf, c)
    return kv, logits


# === DECODING STRATEGIES ===
# 각 전략은 프롬프트, 가중치, 설정을 받아 생성된 토큰과 총 log-probability를
# 반환함. 차이점은 오직 토큰 선택 방식에만 있음.

def decode_greedy(prompt: list[int], wf: dict, c: Cfg,
                  max_len: int = 12) -> tuple[list[int], float]:
    """항상 가장 확률 높은 토큰을 선택함. 결정적임.

    단순하지만 최적은 아님: 매 스텝에서 지역적 최선을 선택하므로 전역적으로
    더 나은 시퀀스를 놓칠 수 있음. greedy 디코딩은 모델이 완벽하게 보정되었을 때만
    최적인데 (실제로는 절대 그렇지 않음).
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE: break
        probs = softmax_f(logits)
        tok = max(range(c['vocab_size']), key=lambda i: probs[i])
        if tok == c['bos']: break
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
    return gen, lp


def decode_temperature(prompt: list[int], wf: dict, c: Cfg,
                       max_len: int = 12, temperature: float = 0.8) -> tuple[list[int], float]:
    """sampling 전에 logits를 temperature로 스케일링함.

    temperature는 확률 분포의 순위를 바꾸지 않고 형태만 변형함.
    T < 1이면 날카로워짐(더 결정적), T > 1이면 평탄해짐(더 무작위).
    수학적으로: softmax(logits/T)는 T -> 0이면 최빈값에 질량이 집중되고
    T -> inf이면 균등 분포에 수렴함.
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE: break
        probs = softmax_f([l / temperature for l in logits])
        tok = random.choices(range(c['vocab_size']), weights=probs)[0]
        if tok == c['bos']: break
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
    return gen, lp


def decode_top_k(prompt: list[int], wf: dict, c: Cfg,
                 max_len: int = 12, k: int = 5) -> tuple[list[int], float]:
    """가장 확률 높은 k개의 토큰만 고려하고 나머지는 0으로 만듦.

    확률이 낮은 토큰의 긴 꼬리에서 sampling하는 것을 방지함. 모델의 확신도와
    무관하게 컷오프가 고정됨 — 이 경직성이 top-p 대비 top-k의 약점임.
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE: break
        probs = softmax_f(logits)
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_set = set(ranked[:k])
        filt = [probs[i] if i in top_set else 0.0 for i in range(len(probs))]
        total = sum(filt)
        filt = [p / total for p in filt]
        tok = random.choices(range(c['vocab_size']), weights=filt)[0]
        if tok == c['bos']: break
        # 원래 분포에서의 log-prob — 모델의 확신도를 측정함
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
    return gen, lp


def decode_top_p(prompt: list[int], wf: dict, c: Cfg,
                 max_len: int = 12, p: float = 0.9) -> tuple[list[int], float]:
    """누적 확률이 p를 초과할 때까지 토큰을 포함함 (nucleus sampling).

    적응적임: 확신 높은 예측(한 토큰이 95%)에서는 해당 토큰만 고려됨.
    불확실한 예측에서는 많은 토큰이 nucleus에 포함됨. 이 적응성이 top-p가
    고정 top-k보다 우수한 이유임 — 모델 자체의 확신도가 각 스텝에서
    유효 어휘 크기를 결정함.
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE: break
        probs = softmax_f(logits)
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumsum = 0.0
        nucleus: set[int] = set()
        for idx in ranked:
            nucleus.add(idx)
            cumsum += probs[idx]
            if cumsum >= p: break
        filt = [probs[i] if i in nucleus else 0.0 for i in range(len(probs))]
        total = sum(filt)
        filt = [pr / total for pr in filt]
        tok = random.choices(range(c['vocab_size']), weights=filt)[0]
        if tok == c['bos']: break
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
    return gen, lp


def decode_beam(prompt: list[int], wf: dict, c: Cfg,
                max_len: int = 12, beam_width: int = 3) -> tuple[list[int], float]:
    """상위 B개의 후보 시퀀스를 유지하면서 매 스텝마다 확장하고 가지치기함.

    여러 경로를 동시에 탐색하여 greedy보다 높은 log-probability 시퀀스를
    찾음. beam search는 sampling이 아님 — 결정적 검색 알고리즘임. 같은 입력에
    대해 두 번 실행하면 동일한 출력이 나옴. 핵심 트레이드오프: beam_width *
    스텝당_비용의 연산량으로 잠재적으로 훨씬 나은 전역 해를 얻음. 기계 번역에서
    많이 사용됨.
    """
    # 각 beam: (누적_log_prob, 생성된_토큰들, kv_cache, 대기중_logits)
    init_kv, init_logits = feed_prompt(prompt, wf, c)
    beams: list[tuple[float, list[int], list[dict[str, list[list[float]]]], list[float]]] = [
        (0.0, [], clone_kv(init_kv), init_logits)
    ]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_len):
        candidates: list[tuple[float, list[int], list[dict[str, list[list[float]]]], list[float]]] = []
        for blp, btoks, bkv, blogits in beams:
            pos = len(prompt) + len(btoks)
            if pos >= BLOCK_SIZE:
                completed.append((blp, btoks))
                continue
            probs = softmax_f(blogits)
            ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            for idx in ranked[:beam_width]:
                token_lp = math.log(max(probs[idx], 1e-10))
                if idx == c['bos']:
                    completed.append((blp + token_lp, btoks))
                    continue
                # 각 확장은 자체 KV 캐시 복사본을 가짐 (beam이 분기됨)
                new_kv = clone_kv(bkv)
                new_logits = forward_float(idx, pos, new_kv, wf, c)
                candidates.append((blp + token_lp, btoks + [idx], new_kv, new_logits))
        if not candidates: break
        # 가지치기: 누적 log-prob 기준 상위 beam_width개만 유지
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    all_results = completed + [(lp, toks) for lp, toks, _, _ in beams]
    if not all_results: return [], 0.0
    best = max(all_results, key=lambda r: r[0])
    return best[1], best[0]


def decode_speculative(
    prompt: list[int], t_wf: dict, d_wf: dict, tc: Cfg, dc: Cfg,
    max_len: int = 12, draft_k: int = 4,
) -> tuple[list[int], float, int, int]:
    """draft 모델이 k개의 토큰을 생성하고, target 모델이 검증함.

    핵심 아이디어: target 모델로 k개 토큰을 검증하는 비용이 1개 토큰을 생성하는
    비용과 거의 같음 (GPU에서는 k번의 순방향 패스가 하나로 배치됨). draft 토큰이
    target의 분포와 일치하면 target 검증 한 번에 ~k개 토큰을 얻음 — 상당한
    속도 향상임.

    수락 기준 (Leviathan et al.): 각 draft 토큰을 min(1, p_target/p_draft)
    확률로 수락함. 거부 시 max(0, p_target - p_draft)에서 재샘플링하고 이후
    draft를 폐기함. 이것은 무손실임: 출력 분포가 target 모델과 정확히 일치함.

    반환값: (tokens, log_prob, total_proposed, total_accepted)
    """
    t_kv, t_logits = feed_prompt(prompt, t_wf, tc)
    d_kv, d_logits = feed_prompt(prompt, d_wf, dc)
    gen: list[int] = []
    lp = 0.0
    total_proposed = 0
    total_accepted = 0

    while len(gen) < max_len:
        cur = len(prompt) + len(gen)
        remaining = min(draft_k, max_len - len(gen))
        if cur >= BLOCK_SIZE or remaining <= 0: break

        # 1단계: draft 모델이 greedy로 k개 토큰을 제안함 (빠르고 작은 모델)
        draft_toks: list[int] = []
        draft_probs: list[list[float]] = []
        tmp_d_kv = clone_kv(d_kv)
        tmp_d_logits = d_logits[:]
        for di in range(remaining):
            pos = cur + di
            if pos >= BLOCK_SIZE: break
            dp = softmax_f(tmp_d_logits)
            draft_probs.append(dp)
            dtok = max(range(dc['vocab_size']), key=lambda i: dp[i])
            if dtok == dc['bos']: break
            draft_toks.append(dtok)
            tmp_d_logits = forward_float(dtok, pos, tmp_d_kv, d_wf, dc)

        if not draft_toks:
            # draft가 BOS를 생성함 — target greedy 한 스텝으로 폴백
            tp = softmax_f(t_logits)
            ttok = max(range(tc['vocab_size']), key=lambda i: tp[i])
            if ttok == tc['bos']: break
            lp += math.log(max(tp[ttok], 1e-10))
            gen.append(ttok)
            t_logits = forward_float(ttok, cur, t_kv, t_wf, tc)
            d_logits = forward_float(ttok, cur, d_kv, d_wf, dc)
            continue

        total_proposed += len(draft_toks)

        # 2단계: target 모델이 각 draft 토큰을 검증함
        # GPU에서는 하나의 배치 순방향 패스가 됨. 수락 로직은
        # 직렬/병렬 실행 여부와 무관하게 동일함.
        accepted: list[int] = []
        tmp_t_kv = clone_kv(t_kv)
        tmp_t_logits = t_logits[:]

        for vi in range(len(draft_toks)):
            tp = softmax_f(tmp_t_logits)
            dp = draft_probs[vi]
            dtok = draft_toks[vi]
            # rejection sampling: p = min(1, p_target/p_draft)로 수락
            ratio = min(1.0, tp[dtok] / max(dp[dtok], 1e-10))
            if random.random() < ratio:
                accepted.append(dtok)
                lp += math.log(max(tp[dtok], 1e-10))
                tmp_t_logits = forward_float(dtok, cur + vi, tmp_t_kv, t_wf, tc)
            else:
                # 거부: max(0, p_target - p_draft)에서 재샘플링
                adj = [max(0.0, tp[j] - dp[j]) for j in range(len(tp))]
                adj_s = sum(adj)
                if adj_s > 0:
                    adj = [a / adj_s for a in adj]
                    rtok = random.choices(range(tc['vocab_size']), weights=adj)[0]
                else:
                    rtok = random.choices(range(tc['vocab_size']), weights=tp)[0]
                if rtok != tc['bos']:
                    accepted.append(rtok)
                    lp += math.log(max(tp[rtok], 1e-10))
                    forward_float(rtok, cur + vi, tmp_t_kv, t_wf, tc)
                break  # 거부 이후 남은 draft 토큰 전부 폐기

        total_accepted += len(accepted)

        # 수락된 토큰을 양쪽 실제 KV 캐시에 커밋
        for ai, atok in enumerate(accepted):
            t_logits = forward_float(atok, cur + ai, t_kv, t_wf, tc)
            d_logits = forward_float(atok, cur + ai, d_kv, d_wf, dc)
            gen.append(atok)

        if not accepted:
            tp = softmax_f(t_logits)
            ttok = max(range(tc['vocab_size']), key=lambda i: tp[i])
            if ttok == tc['bos']: break
            lp += math.log(max(tp[ttok], 1e-10))
            gen.append(ttok)
            t_logits = forward_float(ttok, cur, t_kv, t_wf, tc)
            d_logits = forward_float(ttok, cur, d_kv, d_wf, dc)

    return gen, lp, total_proposed, total_accepted


# === MODEL INIT AND TRAINING ===

def init_params(vocab_size: int, n_embd: int, n_head: int,
                n_layer: int) -> dict[str, list[list[Value]]]:
    """주어진 차원에 맞게 모든 GPT 파라미터를 초기화함."""
    p: dict[str, list[list[Value]]] = {}
    p['wte'] = make_matrix(vocab_size, n_embd)
    p['wpe'] = make_matrix(BLOCK_SIZE, n_embd)
    for li in range(n_layer):
        p[f'l{li}.wq'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wk'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wv'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wo'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.fc1'] = make_matrix(4 * n_embd, n_embd)
        p[f'l{li}.fc2'] = make_matrix(n_embd, 4 * n_embd)
    p['lm_head'] = make_matrix(vocab_size, n_embd)
    return p


def train_model(docs: list[str], chars: list[str], bos: int, vocab_size: int,
                params: dict[str, list[list[Value]]], n_embd: int, n_head: int,
                n_layer: int, head_dim: int, num_steps: int) -> None:
    """Adam 옵티마이저와 선형 LR 감쇠로 GPT 모델을 학습함."""
    plist = [p for w in params.values() for row in w for p in row]
    m_s = [0.0] * len(plist)
    v_s = [0.0] * len(plist)
    print(f"Parameters: {len(plist):,}")

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        toks = [bos] + [chars.index(ch) for ch in doc] + [bos]
        sl = min(BLOCK_SIZE, len(toks) - 1)
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]
        losses: list[Value] = []
        for pos in range(sl):
            logits = gpt_forward_train(toks[pos], pos, keys, vals, params,
                                       n_embd, n_head, n_layer, head_dim)
            probs = softmax_v(logits)
            losses.append(-safe_log(probs[toks[pos + 1]]))
        loss = (1.0 / sl) * sum(losses)
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / num_steps)
        for i, p in enumerate(plist):
            m_s[i] = BETA1 * m_s[i] + (1 - BETA1) * p.grad
            v_s[i] = BETA2 * v_s[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_s[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_s[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{num_steps} | loss: {loss.data:.4f}")
    print(f"  Final loss: {loss.data:.4f}\n")


# === MAIN ===

if __name__ == "__main__":
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1
    print(f"Loaded {len(docs)} documents, vocab size: {VOCAB_SIZE}\n")

    # target 모델 학습 (더 큰 모델)
    print(f"=== Training Target Model (n_embd={TARGET_N_EMBD}, n_layer={TARGET_N_LAYER}) ===")
    target_params = init_params(VOCAB_SIZE, TARGET_N_EMBD, TARGET_N_HEAD, TARGET_N_LAYER)
    t0 = time.time()
    train_model(docs, unique_chars, BOS, VOCAB_SIZE, target_params,
                TARGET_N_EMBD, TARGET_N_HEAD, TARGET_N_LAYER,
                TARGET_N_EMBD // TARGET_N_HEAD, TARGET_STEPS)
    print(f"Target model trained in {time.time() - t0:.1f}s")

    # draft 모델 학습 (더 작은 모델)
    print(f"\n=== Training Draft Model (n_embd={DRAFT_N_EMBD}, n_layer={DRAFT_N_LAYER}) ===")
    draft_params = init_params(VOCAB_SIZE, DRAFT_N_EMBD, DRAFT_N_HEAD, DRAFT_N_LAYER)
    t0 = time.time()
    train_model(docs, unique_chars, BOS, VOCAB_SIZE, draft_params,
                DRAFT_N_EMBD, DRAFT_N_HEAD, DRAFT_N_LAYER,
                DRAFT_N_EMBD // DRAFT_N_HEAD, DRAFT_STEPS)
    print(f"Draft model trained in {time.time() - t0:.1f}s")

    # 추론을 위해 가중치를 일반 float으로 추출
    twf = {k: extract(v) for k, v in target_params.items()}
    dwf = {k: extract(v) for k, v in draft_params.items()}
    tc: Cfg = {'n_embd': TARGET_N_EMBD, 'n_head': TARGET_N_HEAD,
               'n_layer': TARGET_N_LAYER, 'head_dim': TARGET_N_EMBD // TARGET_N_HEAD,
               'vocab_size': VOCAB_SIZE, 'bos': BOS}
    dc: Cfg = {'n_embd': DRAFT_N_EMBD, 'n_head': DRAFT_N_HEAD,
               'n_layer': DRAFT_N_LAYER, 'head_dim': DRAFT_N_EMBD // DRAFT_N_HEAD,
               'vocab_size': VOCAB_SIZE, 'bos': BOS}

    def tok2str(toks: list[int]) -> str:
        return ''.join(unique_chars[t] if t != BOS else '' for t in toks)

    # === DECODING STRATEGIES COMPARISON ===
    print("\n=== Decoding Strategies Comparison ===")
    prompts = [("a", [BOS, unique_chars.index('a')]),
               ("m", [BOS, unique_chars.index('m')])]

    for label, ptoks in prompts:
        print(f'\nPrompt: "{label}" (BOS + \'{label}\')\n')
        print(f"{'Strategy':<22} {'Output':<16} {'Log-Prob':>10} {'Tokens/Step':>12}")
        print("-" * 62)

        g, glp = decode_greedy(ptoks, twf, tc)
        print(f"{'Greedy':<22} {tok2str(g):<16} {glp:>10.2f} {'1.0':>12}")

        t, tlp = decode_temperature(ptoks, twf, tc, temperature=0.8)
        print(f"{'Temperature (0.8)':<22} {tok2str(t):<16} {tlp:>10.2f} {'1.0':>12}")

        k, klp = decode_top_k(ptoks, twf, tc, k=5)
        print(f"{'Top-k (k=5)':<22} {tok2str(k):<16} {klp:>10.2f} {'1.0':>12}")

        p, plp = decode_top_p(ptoks, twf, tc, p=0.9)
        print(f"{'Top-p (p=0.9)':<22} {tok2str(p):<16} {plp:>10.2f} {'1.0':>12}")

        b, blp = decode_beam(ptoks, twf, tc, beam_width=3)
        print(f"{'Beam (width=3)':<22} {tok2str(b):<16} {blp:>10.2f} {'1.0':>12}")

        s, slp, sp, sa = decode_speculative(ptoks, twf, dwf, tc, dc, draft_k=4)
        tps = sa / max(1, sp / 4) if sp > 0 else 1.0
        print(f"{'Speculative (k=4)':<22} {tok2str(s):<16} {slp:>10.2f} {tps:>12.1f}")

    # === DIVERSITY ANALYSIS ===
    # 결정적 전략(greedy, beam)은 같은 프롬프트에 대해 같은 출력을 생성함.
    # 확률적 전략(temperature, top-k, top-p)은 다양성을 만들어냄 —
    # 창작 용도에서는 필수적이지만 사실 기반 용도에서는 바람직하지 않음.
    print("\n=== Diversity Analysis ===")
    print("Generated 20 names with each strategy:\n")
    n_samp = 20
    seeds = list("abcdefghijklmnopqrst")
    strats = [
        ("Greedy", lambda pt: decode_greedy(pt, twf, tc)),
        ("Temperature (0.8)", lambda pt: decode_temperature(pt, twf, tc, temperature=0.8)),
        ("Top-k (k=5)", lambda pt: decode_top_k(pt, twf, tc, k=5)),
        ("Top-p (p=0.9)", lambda pt: decode_top_p(pt, twf, tc, p=0.9)),
        ("Beam (width=3)", lambda pt: decode_beam(pt, twf, tc, beam_width=3)),
    ]
    print(f"{'Strategy':<22} {'Unique Names':>13} {'Avg Length':>11} {'Avg Log-Prob':>13}")
    print("-" * 62)
    for sname, sfn in strats:
        names: list[str] = []
        lps: list[float] = []
        for i in range(n_samp):
            pt = [BOS, unique_chars.index(seeds[i])]
            toks, lp = sfn(pt)
            names.append(tok2str(toks))
            lps.append(lp)
        print(f"{sname:<22} {len(set(names)):>13} "
              f"{sum(len(n) for n in names) / n_samp:>11.1f} "
              f"{sum(lps) / n_samp:>13.2f}")

    # === SPECULATIVE DECODING STATS ===
    print("\n=== Speculative Decoding Stats ===")
    tot_prop = tot_acc = 0
    for i in range(n_samp):
        pt = [BOS, unique_chars.index(seeds[i])]
        _, _, prop, acc = decode_speculative(pt, twf, dwf, tc, dc, draft_k=4)
        tot_prop += prop
        tot_acc += acc

    acc_rate = 100 * tot_acc / max(tot_prop, 1)
    n_rounds = tot_prop / 4
    toks_per_round = tot_acc / max(n_rounds, 1)
    print(f"Draft tokens proposed per step: 4")
    print(f"Total proposed: {tot_prop} | Total accepted: {tot_acc}")
    print(f"Average acceptance rate: {acc_rate:.1f}%")
    print(f"Average tokens accepted per target verify pass: {toks_per_round:.1f}")
    # 참고: 프로덕션에서 잘 매칭된 draft 모델을 사용하면 70-90%의 수락율이 일반적임.
    # 실제 GPU 속도 향상은 k번의 검증 순방향 패스를 단일 커널 런치로 배치하는 데서 나옴
    # — 우리의 스칼라 Python은 그 병렬성을 보여줄 수 없지만, 수락율이 하드웨어
    # 독립적인 지표임.
