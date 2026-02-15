"""
Flash Attention은 표준 attention과 동일한 정확한 attention을 계산하지만,
Q, K, V를 타일 단위로 처리하며 online softmax를 사용해 전체 N*N 스코어 행렬을 메모리에 올리지 않음.
"""
# Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
# with IO-Awareness" (2022). https://arxiv.org/abs/2205.14135
# Also: Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018).

from __future__ import annotations

import math
import random
import time

random.seed(42)

# === CONSTANTS AND CONFIGURATIONS ===

D_HEAD = 16  # head 차원 (d_k = d_v)

# 테스트 설정: (시퀀스 길이, 블록 크기) 쌍.
# 여러 설정으로 단일 크기에서의 우연한 정확성이 아님을 검증함.
VERIFY_CONFIGS: list[tuple[int, int]] = [
    (32, 8),
    (64, 8),
    (64, 16),
    (48, 12),   # 2의 거듭제곱이 아닌 경우로 나머지 처리를 테스트함
    (37, 8),    # N이 block_size로 나누어지지 않는 일반적인 경우
]

# 메모리 비교 테이블용 시퀀스 길이
MEMORY_SEQ_LENS: list[int] = [16, 32, 64, 128, 256]
MEMORY_BLOCK_SIZES: list[int] = [4, 8, 16]

# 블록 크기 효과 테이블용 블록 크기
BLOCK_EFFECT_N = 64
BLOCK_EFFECT_SIZES: list[int] = [4, 8, 16, 32]

# 참고: 여기선 즉시 실행과 읽기 쉬운 출력을 위해 작은 차원을 사용함.
# 프로덕션 Flash Attention은 d=128, N=8192+, block_size=64-256 (GPU SRAM 용량에
# 맞춰 튜닝)에서 동작함. 알고리즘은 동일하고 상수만 다름.


# === HELPER FUNCTIONS ===
# 순수 Python 행렬 연산. NumPy 없음, 트릭 없음 -- 명시적 루프만 사용해서
# 모든 메모리 할당이 보이고 셀 수 있음.

def rand_matrix(rows: int, cols: int) -> list[list[float]]:
    """1/sqrt(cols) 스케일링을 적용한 랜덤 행렬로 내적을 O(1)로 유지함.

    스케일링 없으면 QK^T 내적이 d에 비례해서 커져서 softmax가
    포화 상태(거의 one-hot)에 빠짐. Xavier 유사 초기화가 이를 방지함."""
    s = 1.0 / math.sqrt(cols)
    return [[random.gauss(0.0, s) for _ in range(cols)] for _ in range(rows)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """A[m,k] @ B[k,n] -> C[m,n]."""
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    # B를 전치해서 내부 루프가 연속된 행에 접근하게 함 (C에서는 캐시 친화적;
    # Python에서는 무의미하지만, Flash Attention이 GPU에서 활용하는 패턴을 반영함)
    bt = [[b[r][c] for r in range(k)] for c in range(n)]
    return [
        [sum(a[i][p] * bt[j][p] for p in range(k)) for j in range(n)]
        for i in range(m)
    ]


def transpose(mat: list[list[float]]) -> list[list[float]]:
    rows = len(mat)
    cols = len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def softmax_rows(mat: list[list[float]]) -> list[list[float]]:
    """수치 안정성을 위해 행 최댓값을 빼는 행 단위 softmax.

    softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
    max(x)를 빼면 exp() 오버플로를 방지하면서 분포는 보존됨.
    이건 "two-pass" softmax임: 패스 1에서 최댓값을 찾고, 패스 2에서 exp와 합을 계산함."""
    result: list[list[float]] = []
    for row in mat:
        mx = max(row)
        exps = [math.exp(x - mx) for x in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result


def max_abs_diff(a: list[list[float]], b: list[list[float]]) -> float:
    """두 행렬 간의 원소별 최대 절대 차이."""
    return max(
        abs(a[i][j] - b[i][j])
        for i in range(len(a))
        for j in range(len(a[0]))
    )


# === STANDARD ATTENTION ===
# Flash Attention이 대체하는 교과서적 공식. 이걸 계산하려면
# 전체 N*N 스코어 행렬을 메모리에 올려야 함 -- 이게 병목임.

def standard_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]]
) -> tuple[list[list[float]], int]:
    """전체 N*N 스코어 행렬을 메모리에 올려서 attention을 계산함.

    단계:
      S = Q @ K^T / sqrt(d)    -- 스코어 행렬 [N, N]
      P = softmax(S, axis=-1)   -- attention 가중치 [N, N], 행의 합이 1
      O = P @ V                 -- 출력 [N, d]

    최대 메모리: S에 N*N개 float (또는 P -- 같은 크기라 S를 제자리에서 덮어쓸 수 있음).
    이 O(N^2) 메모리가 긴 시퀀스에서 표준 attention이 터지는 이유임.
    N=128K에서 float16으로 스코어 행렬만 32 GB임.

    (output, peak_memory_floats)를 반환함."""
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)

    # S = Q @ K^T -- 메모리에 올리지 않으려는 N*N 행렬
    scores = matmul(q, transpose(k))

    # softmax 전에 스케일링 (Q를 미리 스케일링하는 것과 동등하지만 더 명확함)
    scores = [[v * scale for v in row] for row in scores]

    # P = softmax(S) -- 여전히 N*N
    weights = softmax_rows(scores)

    # O = P @ V -- 다시 [N, d]로
    output = matmul(weights, v)

    # 최대 메모리: N*N 스코어/가중치 행렬
    peak_memory = n * n
    return output, peak_memory


# === FLASH ATTENTION ===
#
# Dao et al.의 핵심 통찰: attention을 전체 N*N 스코어 행렬을 저장하지 않고도
# 타일 단위로 계산할 수 있음. 핵심 트릭은 타일들에 걸쳐 실행 중 통계(최댓값과
# 분모 합)를 유지하는 "online softmax"임.
#
# GPU에서 tiling이 중요한 이유 (이 시뮬레이션에서는 해당 없음):
#   GPU 메모리는 두 계층임: HBM (크고 느림)과 SRAM (작고 빠름).
#   표준 attention은 Q,K를 HBM에서 읽고, N*N 스코어를 HBM에 쓰고, softmax를 위해
#   다시 읽고, 가중치를 HBM에 쓰고, P@V를 위해 다시 읽음.
#   Flash Attention은 Q,K,V의 타일을 SRAM에 로드하고, SRAM 내에서 attention을 계산하고,
#   최종 출력만 HBM에 씀. 총 HBM 읽기가 O(N^2)에서 O(N)으로 줄어듦.
#
# 이 시뮬레이션은 이를 가능하게 하는 알고리즘(tiling + online softmax)을 보여줌.
# 속도 향상은 보여주지 않음 -- 그건 GPU 메모리 계층 구조에서 나오는 것임.
#
# === ONLINE SOFTMAX: THE CORE INSIGHT ===
#
# 표준 softmax는 분모를 계산하기 위해 모든 스코어가 필요함:
#   softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
#
# Online softmax는 스코어를 블록 단위로 처리하며 실행 중 통계를 유지함:
#   m = 실행 중 최댓값 (수치 안정성용)
#   l = exp(score - m)의 실행 중 합 (softmax 분모)
#
# 새 블록이 로컬 최댓값 m_new와 함께 도착하면:
#   1. m_combined = max(m_old, m_new)
#   2. 이전 합 재조정:    l_old' = l_old * exp(m_old - m_combined)
#   3. 새 블록 계산:      l_new  = sum(exp(scores - m_combined))
#   4. l_combined = l_old' + l_new
#   5. 이전 출력 재조정:  O' = O * (l_old / l_combined) * exp(m_old - m_combined)
#      (최댓값과 분모의 재조정을 한 단계로 병합)
#   6. 새 기여분 추가:    O' += (1/l_combined) * exp(scores - m_combined) @ V_block
#
# 모든 블록 처리 후, O는 표준 attention과 정확히 같은 결과를 가짐.
# 수학적 증명: 재조정 체인이 텔레스코핑됨 -- 각 단계에서 이전의 모든 기여분에
# 동일한 보정 계수가 곱해져서, 지금까지 본 모든 블록에 걸친 exp(score_i)와
# 총합의 비율이 보존됨.

def flash_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]],
    block_size: int,
) -> tuple[list[list[float]], int]:
    """Flash Attention: N*N 행렬을 메모리에 올리지 않고 정확한 attention을 계산함.

    Q, K, V를 block_size 크기의 타일로 처리함. 각 쿼리 블록에 대해 모든
    키/값 블록을 순회하며, online softmax로 올바른 정규화를 유지하면서
    모든 스코어를 저장하지 않고 출력을 누적함.

    최대 메모리: block_size * block_size개 float (한 번에 스코어 타일 하나).
    표준 attention의 N*N과 비교해볼 것.

    (output, peak_memory_floats)를 반환함."""
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)

    # 쿼리별 실행 중 통계. 각 쿼리 행이 자체 최댓값과 합을 가짐 -- softmax가
    # 행별로 독립적으로 적용되기 때문 (각 쿼리가 개별적으로 attend함).
    output = [[0.0] * d for _ in range(n)]
    row_max = [float("-inf")] * n   # m_i: 쿼리 i의 스코어 실행 중 최댓값
    row_sum = [0.0] * n             # l_i: 쿼리 i의 exp(score - m_i) 실행 중 합

    peak_memory = 0

    # 외부 루프: 쿼리 블록 단위로 순회
    for q_start in range(0, n, block_size):
        q_end = min(q_start + block_size, n)
        q_block = q[q_start:q_end]
        bq = q_end - q_start  # 실제 블록 크기 (경계에서는 더 작을 수 있음)

        # 내부 루프: 각 쿼리 블록에 대해 모든 키/값 블록을 순회함.
        # 이게 "tiling"임 -- 한 번에 bq * bk 타일 하나의 스코어만 존재함.
        for k_start in range(0, n, block_size):
            k_end = min(k_start + block_size, n)
            k_block = k[k_start:k_end]
            v_block = v[k_start:k_end]
            bk = k_end - k_start

            # 시뮬레이션된 메모리 추적: 스코어 타일이 가장 큰 임시 메모리
            peak_memory = max(peak_memory, bq * bk)

            # 단계 1: 부분 스코어 S_ij = Q_block @ K_block^T / sqrt(d) 계산
            # bq * bk 행렬임 -- N*N이 아님.
            scores_tile: list[list[float]] = []
            for qi in range(bq):
                row: list[float] = []
                for ki in range(bk):
                    dot = sum(q_block[qi][c] * k_block[ki][c] for c in range(d))
                    row.append(dot * scale)
                scores_tile.append(row)

            # 단계 2: 이 블록의 각 쿼리 행에 대해 online softmax 업데이트 적용
            for qi in range(bq):
                global_i = q_start + qi  # 전체 출력에서의 인덱스

                # 이 타일 행의 로컬 최댓값
                #   m_ij = max(S_ij[qi, :])
                m_tile = max(scores_tile[qi])

                # 결합된 최댓값: 실행 중 최댓값과 이 타일 최댓값의 max
                #   m_new = max(m_old, m_tile)
                m_old = row_max[global_i]
                m_new = max(m_old, m_tile)

                # 이전 누적기의 재조정 계수:
                #   최댓값이 증가하면 이전의 모든 exp() 값이 이전 최댓값 기준으로
                #   계산된 것임. exp(m_old - m_new)를 곱하면 새 최댓값 기준으로 보정됨.
                #   exp(score - m_old) * exp(m_old - m_new) = exp(score - m_new)
                if m_old == float("-inf"):
                    # 이 쿼리의 첫 번째 블록 -- 재조정할 이전 누적기가 없음
                    old_scale = 0.0
                else:
                    old_scale = math.exp(m_old - m_new)

                # 이 타일 행의 각 스코어에 대해 exp(score - m_new) 계산
                #   P_ij[qi, ki] = exp(S_ij[qi, ki] - m_new)
                exp_scores = [math.exp(s - m_new) for s in scores_tile[qi]]

                # 새 지수화된 스코어의 합: 분모에 대한 기여분
                new_sum = sum(exp_scores)

                # 실행 중 분모 업데이트:
                #   l_new = l_old * exp(m_old - m_new) + sum(exp(scores - m_new))
                l_old = row_sum[global_i]
                l_new = l_old * old_scale + new_sum

                # 출력 누적기 업데이트:
                #   O_i = O_i * (l_old * old_scale / l_new) + (1/l_new) * P_ij @ V_block
                #
                # 첫 번째 항은 이전 출력을 재조정함 (새 최댓값 보정과 업데이트된
                # 분모로 재가중). 두 번째 항은 이 타일의 가중된 기여분을 추가하며
                # 이미 정규화되어 있음.
                if l_new > 0.0:
                    # 이전 누적기 재조정
                    rescale = (l_old * old_scale) / l_new
                    for c in range(d):
                        output[global_i][c] *= rescale

                    # 새 기여분 추가: (1/l_new) * sum_ki(exp_scores[ki] * V[ki, c])
                    inv_l = 1.0 / l_new
                    for ki in range(bk):
                        w = exp_scores[ki] * inv_l
                        for c in range(d):
                            output[global_i][c] += w * v_block[ki][c]

                # 실행 중 통계 업데이트
                row_max[global_i] = m_new
                row_sum[global_i] = l_new

    # 최종 정규화가 필요 없음 -- 각 단계에서 l(실행 중 분모)로 나누기 때문에
    # 출력이 이미 올바르게 정규화되어 있음. 정규화를 마지막으로 미루는 일부
    # 표현과 다름; 여기선 각 내부 반복 후 출력이 항상 "지금까지 완전히 정규화된" 상태임.

    return output, peak_memory


# === VERIFICATION ===

def verify(
    n: int, d: int, block_size: int, tolerance: float = 1e-6
) -> tuple[bool, float, int, int]:
    """동일한 입력에 대해 표준 및 flash attention을 실행하고 출력이 일치하는지 확인함.

    (passed, max_diff, standard_memory, flash_memory)를 반환함."""
    q = rand_matrix(n, d)
    k = rand_matrix(n, d)
    v = rand_matrix(n, d)

    out_std, mem_std = standard_attention(q, k, v)
    out_flash, mem_flash = flash_attention(q, k, v, block_size)

    diff = max_abs_diff(out_std, out_flash)
    passed = diff < tolerance
    return passed, diff, mem_std, mem_flash


# === MEMORY ANALYSIS ===
# 표준 attention: 최대 메모리 = N^2 (전체 스코어 행렬).
# Flash attention: 최대 메모리 = B^2 (스코어 타일 하나), N과 무관함.
#
# 실제 GPU에서 Flash Attention은 HBM I/O도 O(N^2 * d)에서
# O(N * d^2 / SRAM_size)로 줄이지만, 그건 메모리 계층 구조에 대한
# I/O 복잡도 논의임 -- 순수 Python으로는 시뮬레이션 불가능함.

def format_int(n: int) -> str:
    """정수를 쉼표 구분자로 포맷함."""
    return f"{n:,}"


def print_memory_table(seq_lens: list[int], block_sizes: list[int]) -> None:
    """다양한 설정에서 표준 vs flash의 최대 메모리 비교를 출력함."""
    # 헤더 구성
    header = f"{'Seq Length (N)':>14}   {'Standard (floats)':>18}"
    for b in block_sizes:
        header += f"   {'Flash B=' + str(b):>12}"
    print(header)

    separator = "\u2500" * 14 + "   " + "\u2500" * 18
    for _ in block_sizes:
        separator += "   " + "\u2500" * 12
    print(separator)

    for n in seq_lens:
        std_mem = n * n
        row = f"{n:>14}   {format_int(std_mem):>18}"
        for b in block_sizes:
            flash_mem = b * b
            row += f"   {format_int(flash_mem):>12}"
        print(row)


def print_block_effect_table(n: int, d: int, block_sizes: list[int]) -> None:
    """블록 크기가 메모리와 타일 수에 미치는 영향을 보여줌."""
    header = f"{'Block Size':>10}   {'Memory (floats)':>15}   {'Num Tiles':>9}"
    print(header)
    separator = "\u2500" * 10 + "   " + "\u2500" * 15 + "   " + "\u2500" * 9
    print(separator)

    for b in block_sizes:
        mem = b * b
        # 타일 수: ceil(N/B) 쿼리 블록 * ceil(N/B) 키 블록
        num_q_blocks = math.ceil(n / b)
        num_k_blocks = math.ceil(n / b)
        num_tiles = num_q_blocks * num_k_blocks
        print(f"{b:>10}   {format_int(mem):>15}   {num_tiles:>9}")


# === MAIN ===

if __name__ == "__main__":
    print("=== Flash Attention: Algorithmic Simulation ===\n")

    # 참고: 시뮬레이션 vs 최적화의 구분을 즉시 명확하게 함
    print("Signpost: This is an algorithmic simulation, not a performance benchmark.")
    print("Pure Python is slower than standard attention here. The point is showing WHAT")
    print("Flash Attention does (tiled computation, online softmax), not achieving speedup.")
    print("On GPU, the speedup comes from keeping tiles in SRAM (fast, small) instead of")
    print("reading/writing the N*N matrix from HBM (large, slow).\n")

    # --- Verification ---
    print("--- Verification ---")
    all_passed = True

    for n, block_size in VERIFY_CONFIGS:
        print(f"\nConfig: N={n}, d={D_HEAD}, block_size={block_size}")
        t0 = time.time()
        passed, diff, mem_std, mem_flash = verify(n, D_HEAD, block_size)
        elapsed = time.time() - t0

        print(f"  Standard attention: computed (peak memory: {format_int(mem_std)} floats)")
        print(f"  Flash attention:    computed (peak memory: {format_int(mem_flash)} floats)")
        print(f"  Max element difference: {diff:.2e}")
        print(f"  Time: {elapsed*1000:.1f} ms")

        if passed:
            print(f"  PASS: outputs match within 1e-6 tolerance")
        else:
            print(f"  FAIL: outputs diverge beyond 1e-6 tolerance")
            all_passed = False

    print(f"\nOverall: {'all configurations passed' if all_passed else 'SOME CONFIGURATIONS FAILED'}")

    # --- Memory Comparison ---
    # 이 테이블이 핵심 결과임: 표준 attention 메모리는 O(N^2)로 증가하지만
    # flash attention 메모리는 시퀀스 길이와 무관하게 O(B^2)에 머무름.
    print("\n--- Memory Comparison ---")
    print("Peak floats allocated for the score matrix (standard) vs one tile (flash):\n")
    print_memory_table(MEMORY_SEQ_LENS, MEMORY_BLOCK_SIZES)

    print(f"\nStandard attention memory grows as O(N^2) -- doubling N quadruples memory.")
    print(f"Flash attention memory is O(B^2), independent of sequence length N.")
    print(f"At N=128K with B=128, standard needs 16 billion floats; flash needs 16,384.")

    # --- Block Size Effect ---
    # 블록이 작으면 = 메모리는 적지만 타일이 많아짐 (루프 반복 증가).
    # GPU에서는 블록이 작으면 SRAM 로드가 많아짐 -- SRAM을 낭비 없이
    # 채우는 최적 지점이 있음.
    print(f"\n--- Block Size Effect ---")
    print(f"For N={BLOCK_EFFECT_N}, d={D_HEAD}:\n")
    print_block_effect_table(BLOCK_EFFECT_N, D_HEAD, BLOCK_EFFECT_SIZES)

    print(f"\nSmaller blocks use less memory but require more tiles (iterations).")
    print(f"On GPU, the optimal block size fills SRAM: A100 has 192KB SRAM,")
    print(f"fitting B~128 for d=128 in float16. Pure Python has no SRAM,")
    print(f"so block size affects only iteration count here.")

    # 참고: 런타임 벤치마크 섹션 없음. Flash Attention의 속도 향상은 FLOP 감소가 아니라
    # GPU 메모리 계층 구조(SRAM vs HBM)에서 나옴. 순수 Python에서는 연산당 인터프리터
    # 오버헤드가 지배적이라 flash가 표준보다 더 느림. GPU에서는 Flash Attention이
    # HBM 읽기를 O(N^2)에서 O(N)으로 줄여서 2-4배 더 빠름.
    # 위의 메모리 비교 테이블이 여기서 의미 있는 결과임.
