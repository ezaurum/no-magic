"""
중요한 attention 메커니즘을 나란히 비교함: MHA, GQA, MQA, sliding window이
동일한 입력에서 메모리, 연산, 표현력을 어떻게 트레이드오프하는지 보여줌.
"""
# Reference: Vaswani et al., "Attention Is All You Need" (2017) for scaled dot-product
# and multi-head attention. Shazeer, "Fast Transformer Decoding" (2019) for multi-query.
# Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from
# Multi-Head Checkpoints" (2023). Beltagy et al., "Longformer" (2020) for sliding window.

from __future__ import annotations

import math
import random
import time

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===

SEQ_LEN = 32
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # = 16
N_KV_HEADS_GQA = 2
WINDOW_SIZE = 8

# 참고: 프로덕션 트랜스포머는 d_model=4096+, 32-128 헤드, seq_len=8192+를 사용함.
# 이 토이 차원은 모든 알고리즘 디테일을 보존하면서 빠르게 실행됨.


# === HELPER FUNCTIONS ===
# 일반 Python 리스트의 리스트로 구현한 행렬 연산 -- attention의 기반이 되는
# 선형대수 프리미티브들임.

def rand_matrix(rows: int, cols: int) -> list[list[float]]:
    """Xavier 스타일 1/sqrt(cols) 스케일링으로 softmax 포화를 방지하는 랜덤 행렬."""
    s = 1.0 / math.sqrt(cols)
    return [[random.gauss(0, s) for _ in range(cols)] for _ in range(rows)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """A[m,k] @ B[k,n] -> C[m,n]. 모든 attention 변형에서 가장 비용이 큰 연산임."""
    k = len(a[0])
    n = len(b[0])
    # B를 미리 전치해서 내부 루프의 행 접근이 연속적이게 함
    bt = [[b[r][c] for r in range(k)] for c in range(n)]
    return [[sum(a[i][p] * bt[j][p] for p in range(k)) for j in range(n)]
            for i in range(len(a))]


def transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]


def softmax_row(row: list[float]) -> list[float]:
    """안정적인 softmax: max를 빼서 exp() 오버플로를 방지함.
    exp(x-c)/sum(exp(x_j-c)) = exp(x)/sum(exp(x_j))이 임의의 상수 c에 대해 성립함."""
    mx = max(row)
    exps = [math.exp(x - mx) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def flatten(m: list[list[float]]) -> list[float]:
    return [v for row in m for v in row]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """벡터 간 방향 일치도를 측정함. 크기는 무시함."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0


def avg_head_weights(
    w: list[list[float]], head_dim: int, n_heads: int, n_kv_heads: int
) -> list[list[float]]:
    """MHA 헤드 그룹을 평균내서 축소된 KV 가중치를 만듦 (Ainslie et al. 2023).
    MHA에서 GQA/MQA로 변환할 때, 각 그룹 내 KV 열을 평균 풀링함."""
    gs = n_heads // n_kv_heads
    d = len(w)
    kv_dim = n_kv_heads * head_dim
    result = [[0.0] * kv_dim for _ in range(d)]
    for r in range(d):
        for g in range(n_kv_heads):
            for c in range(head_dim):
                result[r][g * head_dim + c] = sum(
                    w[r][(g * gs + h) * head_dim + c] for h in range(gs)
                ) / gs
    return result


# === ATTENTION VARIANTS ===

def vanilla_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]]
) -> list[list[float]]:
    """Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V

    sqrt(d_k) 스케일링이 핵심임: 이게 없으면 내적이 d_k에 비례해서 커지고,
    softmax가 포화 상태(거의 원-핫 출력)로 밀림. sqrt(d_k)로 나누면
    분산이 ~1.0으로 유지돼서 softmax가 유용한 분포를 만듦."""
    scale = 1.0 / math.sqrt(len(q[0]))
    # scores[i][j] = 위치 i가 위치 j에 얼마나 attend하는지
    scores = [[v * scale for v in row] for row in matmul(q, transpose(k))]
    weights = [softmax_row(row) for row in scores]
    return matmul(weights, v)


def multi_head_attention(
    x: list[list[float]], w_q: list[list[float]], w_k: list[list[float]],
    w_v: list[list[float]], w_o: list[list[float]], n_heads: int,
) -> list[list[float]]:
    """MHA(X) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(X @ W_Q_i, X @ W_K_i, X @ W_V_i)

    왜 여러 헤드인가? 각 헤드가 다른 종류의 "관련성"을 학습함 --
    구문적, 의미적, 위치적. 전체 attention FLOPs은 단일 헤드와 동일함;
    d_model을 헤드 수로 분할할 뿐임. 이점은 표현력에 있음."""
    seq_len = len(x)
    hd = len(x[0]) // n_heads
    q_full, k_full, v_full = matmul(x, w_q), matmul(x, w_k), matmul(x, w_v)

    heads = []
    for h in range(n_heads):
        lo, hi = h * hd, (h + 1) * hd
        heads.append(vanilla_attention(
            [r[lo:hi] for r in q_full], [r[lo:hi] for r in k_full],
            [r[lo:hi] for r in v_full]))

    # 헤드를 연결한 뒤 프로젝션함. W_O가 헤드 간 믹싱이 일어나는 곳임.
    concat = [[c for h in heads for c in h[i]] for i in range(seq_len)]
    return matmul(concat, w_o)


def grouped_query_attention(
    x: list[list[float]], w_q: list[list[float]], w_k_r: list[list[float]],
    w_v_r: list[list[float]], w_o: list[list[float]],
    n_heads: int, n_kv_heads: int,
) -> list[list[float]]:
    """GQA: query 헤드를 그룹으로 나눠서 KV 프로젝션을 공유함.
    4개 query 헤드와 2개 KV 헤드가 있으면, 헤드 0-1이 하나의 KV 쌍을, 2-3이 다른 쌍을 공유함.

    왜 이게 되는가: 학습된 MHA 모델에서 KV 표현은 헤드 간에 매우 상관관계가 높음.
    GQA는 이 중복성을 활용함 -- KV 헤드를 공유해도 품질 손실이 적으면서
    KV cache 메모리를 비례적으로 줄임.
    LLaMA 2 70B는 64개 query 헤드에 8개 KV 헤드로 GQA를 사용함."""
    seq_len = len(x)
    hd = len(x[0]) // n_heads
    gs = n_heads // n_kv_heads  # KV 그룹당 query 헤드 수

    q_full = matmul(x, w_q)
    k_r, v_r = matmul(x, w_k_r), matmul(x, w_v_r)

    heads = []
    for h in range(n_heads):
        q_lo, q_hi = h * hd, (h + 1) * hd
        g = h // gs  # KV 그룹 인덱스
        kv_lo, kv_hi = g * hd, (g + 1) * hd
        heads.append(vanilla_attention(
            [r[q_lo:q_hi] for r in q_full], [r[kv_lo:kv_hi] for r in k_r],
            [r[kv_lo:kv_hi] for r in v_r]))

    concat = [[c for h in heads for c in h[i]] for i in range(seq_len)]
    return matmul(concat, w_o)


def multi_query_attention(
    x: list[list[float]], w_q: list[list[float]], w_k_s: list[list[float]],
    w_v_s: list[list[float]], w_o: list[list[float]], n_heads: int,
) -> list[list[float]]:
    """MQA: 모든 query 헤드가 하나의 KV 헤드를 공유함 (n_kv=1인 GQA).

    autoregressive 디코딩 시 KV cache는 O(layers*heads*seq*hd)로 스케일됨.
    MQA는 이걸 O(layers*seq*hd)로 줄임 -- 헤드 수만큼 감소함.
    PaLM, Falcon, StarCoder가 이 메모리 절약을 위해 MQA를 사용함.
    트레이드오프: 모든 헤드가 하나의 KV 뷰를 보기 때문에 표현 다양성이 줄어듦."""
    seq_len = len(x)
    hd = len(x[0]) // n_heads

    q_full = matmul(x, w_q)
    k_s, v_s = matmul(x, w_k_s), matmul(x, w_v_s)  # [seq_len, head_dim]

    heads = []
    for h in range(n_heads):
        lo, hi = h * hd, (h + 1) * hd
        # 모든 헤드가 같은 K, V를 재사용함 -- MQA의 핵심 아이디어
        heads.append(vanilla_attention([r[lo:hi] for r in q_full], k_s, v_s))

    concat = [[c for h in heads for c in h[i]] for i in range(seq_len)]
    return matmul(concat, w_o)


def sliding_window_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]],
    window_size: int,
) -> list[list[float]]:
    """각 위치가 가장 가까운 w개의 이전 위치에만 attend함: O(n^2)가 아닌 O(n*w).

    왜 로컬 attention이 되는가: 대부분의 관련 컨텍스트는 가까이에 있음.
    100K 토큰에 대한 풀 attention은 먼 곳의 관련 없는 위치에 연산을 낭비함.

    참고: Mistral 7B는 sliding window (w=4096)을 풀 attention 레이어와
    번갈아 사용함. Longformer는 희소 글로벌 토큰을 추가함. 여기선 순수 로컬만 구현함."""
    d_k = len(q[0])
    scale = 1.0 / math.sqrt(d_k)
    output = []
    for i in range(len(q)):
        start = max(0, i - window_size + 1)
        # 윈도우 내에서만 점수를 계산함 -- 윈도우 밖의 위치는 마스킹이 아니라
        # 아예 계산하지 않음. 이게 연산 절약의 핵심임.
        scores = [sum(q[i][d] * k[j][d] for d in range(d_k)) * scale
                  for j in range(start, i + 1)]
        weights = softmax_row(scores)
        row = [0.0] * d_k
        for idx, j in enumerate(range(start, i + 1)):
            for d in range(d_k):
                row[d] += weights[idx] * v[j][d]
        output.append(row)
    return output


# === FLOP AND MEMORY ANALYSIS ===
# FLOPs: 곱셈-덧셈 = 2 FLOPs. 메모리: 점수 / KV cache의 피크 float 수.
#
# 핵심 공식:
#   Vanilla:  4*n^2*d  (QK^T + attn@V, 프로젝션 제외)
#   MHA:      8*n*d^2 + 4*n^2*d  (프로젝션 + attention)
#   GQA:      KV 프로젝션 절약 (K,V에 대해 2*n*d*d 대신 2*n*d*(nkv*hd))
#   MQA:      최소 KV 프로젝션 (K,V에 대해 2*n*d*hd)
#   Window:   8*n*d^2 + 4*n*w*d  (프로젝션 + 로컬 attention)

def compute_analysis(
    n: int, d: int, h: int, hd: int, nkv: int, w: int
) -> list[tuple[str, int, int]]:
    """각 변형의 (name, flops, memory)를 반환함."""
    return [
        ("Vanilla (single-head)",
         4 * n * n * d,
         n * n),  # 전체 n x n 점수 행렬

        (f"Multi-Head ({h} heads)",
         8 * n * d * d + 4 * n * n * d,
         n * n),  # 순차적: 한 번에 하나의 헤드 점수만

        (f"GQA ({h}q, {nkv}kv heads)",
         2*n*d*d + 2*2*n*d*(nkv*hd) + 2*n*d*d + 4*n*n*d,
         2 * nkv * n * hd),  # KV cache: 주요 절약 포인트

        (f"MQA ({h}q, 1kv head)",
         2*n*d*d + 2*2*n*d*hd + 2*n*d*d + 4*n*n*d,
         2 * n * hd),  # 단일 KV 헤드

        (f"Sliding Window (w={w})",
         8 * n * d * d + 4 * n * w * d,
         n * w),  # 위치당 n 대신 w개의 점수
    ]


# === MAIN: RUN ALL VARIANTS AND COMPARE ===

if __name__ == "__main__":
    print("=== Attention Variants Comparison ===\n")
    print(f"Config: seq_len={SEQ_LEN}, d_model={D_MODEL}, n_heads={N_HEADS}, "
          f"head_dim={HEAD_DIM}, n_kv_heads_gqa={N_KV_HEADS_GQA}, "
          f"window_size={WINDOW_SIZE}\n")

    x = rand_matrix(SEQ_LEN, D_MODEL)
    w_q = rand_matrix(D_MODEL, D_MODEL)
    w_k = rand_matrix(D_MODEL, D_MODEL)
    w_v = rand_matrix(D_MODEL, D_MODEL)
    w_o = rand_matrix(D_MODEL, D_MODEL)

    # MHA 헤드 그룹을 평균내서 축소된 KV 가중치를 만듦 (Ainslie et al. 2023).
    # 이것은 GQA 모델이 MHA 체크포인트에서 초기화되는 방식을 따라함.
    # GQA/MQA 출력이 무관한 랜덤 프로젝션이 아니라 MHA를 근사하게 함.
    w_k_gqa = avg_head_weights(w_k, HEAD_DIM, N_HEADS, N_KV_HEADS_GQA)
    w_v_gqa = avg_head_weights(w_v, HEAD_DIM, N_HEADS, N_KV_HEADS_GQA)
    w_k_mqa = avg_head_weights(w_k, HEAD_DIM, N_HEADS, 1)
    w_v_mqa = avg_head_weights(w_v, HEAD_DIM, N_HEADS, 1)

    results: dict[str, tuple[list[list[float]], float]] = {}

    def run(name: str, fn, *args) -> None:
        t0 = time.time()
        out = fn(*args)
        results[name] = (out, (time.time() - t0) * 1000)

    run("Vanilla (single-head)", vanilla_attention, x, x, x)
    run(f"Multi-Head ({N_HEADS} heads)",
        multi_head_attention, x, w_q, w_k, w_v, w_o, N_HEADS)
    run(f"GQA ({N_HEADS}q, {N_KV_HEADS_GQA}kv heads)",
        grouped_query_attention, x, w_q, w_k_gqa, w_v_gqa, w_o,
        N_HEADS, N_KV_HEADS_GQA)
    run(f"MQA ({N_HEADS}q, 1kv head)",
        multi_query_attention, x, w_q, w_k_mqa, w_v_mqa, w_o, N_HEADS)

    # Sliding window: 풀 MHA 가중치로 프로젝션한 뒤, attention을 로컬 윈도우로
    # 제한함. 프로젝션 차이가 아닌 윈도우 효과만 분리해서 보기 위함임.
    q_p, k_p, v_p = matmul(x, w_q), matmul(x, w_k), matmul(x, w_v)
    t0 = time.time()
    sw = sliding_window_attention(q_p, k_p, v_p, WINDOW_SIZE)
    sw_out = matmul(sw, w_o)
    results[f"Sliding Window (w={WINDOW_SIZE})"] = (sw_out, (time.time() - t0) * 1000)

    # 검증: 모든 출력에 NaN이나 Inf가 없는지 확인
    all_valid = True
    for name, (out, _) in results.items():
        flat = flatten(out)
        if any(math.isnan(v) or math.isinf(v) for v in flat):
            print(f"  WARNING: {name} has numerical issues")
            all_valid = False
    print(f"Numerical validity: {'all outputs clean' if all_valid else 'ISSUES DETECTED'}\n")

    # MHA 대비 코사인 유사도 -- 각 변형이 풀 multi-head attention 대비
    # 정보를 얼마나 보존하는지 측정함
    mha_key = f"Multi-Head ({N_HEADS} heads)"
    mha_flat = flatten(results[mha_key][0])
    sims = {n: cosine_sim(flatten(o), mha_flat) for n, (o, _) in results.items()}

    # 분석적 비용 모델
    analysis = compute_analysis(SEQ_LEN, D_MODEL, N_HEADS, HEAD_DIM,
                                N_KV_HEADS_GQA, WINDOW_SIZE)

    # --- 비교 테이블 출력 ---
    hdr = (f"{'Variant':<28} {'FLOPs':>12} {'Memory':>10} "
           f"{'Cos Sim':>10} {'Time(ms)':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name, flops, mem in analysis:
        cs = sims.get(name, 0.0)
        ms = results[name][1] if name in results else 0.0
        print(f"{name:<28} {flops:>12,} {mem:>10,} {cs:>10.4f} {ms:>10.2f}")

    # --- 핵심 요약 ---
    print("\n=== Key Takeaways ===\n")
    print("1. MHA and vanilla share attention FLOPs (4*n^2*d). MHA adds projection")
    print("   cost (8*n*d^2) but gains multiple learned attention patterns.")
    print(f"2. GQA cuts KV memory {N_HEADS // N_KV_HEADS_GQA}x "
          f"({N_HEADS}->{N_KV_HEADS_GQA} KV heads), output stays close to MHA.")
    print(f"3. MQA cuts KV memory {N_HEADS}x ({N_HEADS}->1 KV head) -- max savings,")
    print("   more quality loss because all heads share one KV view.")
    print(f"4. Sliding window (w={WINDOW_SIZE}) makes attention O(n*w) not O(n^2),")
    print(f"   {SEQ_LEN // WINDOW_SIZE}x cheaper at seq_len={SEQ_LEN}. "
          "Works when locality dominates.")
    print("\nProduction systems compose these: Mistral uses sliding window + GQA,")
    print("LLaMA 2 uses GQA, PaLM/Falcon use MQA. Choose based on whether your")
    print("bottleneck is compute (window), memory (MQA/GQA), or neither (full MHA).")
