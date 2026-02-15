"""
텍스트가 숫자로 변하는 과정 -- 모든 LLM 안에 숨어 있는 압축 알고리즘.
Byte-Pair Encoding은 가장 빈번한 인접 토큰 쌍을 반복적으로 병합하여 어휘를 학습하고,
새로운 텍스트를 인코딩할 때 학습된 병합을 우선순위 순서대로 재생함.
"""
# Reference: Philip Gage, "A New Algorithm for Data Compression" (1994).
# GPT-2's byte-level BPE variant (Radford et al., 2019) starts from raw bytes
# rather than characters -- that's the version implemented here.
# GPT-2의 바이트 수준 BPE 변형(Radford et al., 2019)은 문자가 아닌 원시 바이트에서
# 시작함 -- 여기서 구현한 것이 그 버전임.

from __future__ import annotations

import os
import random
import urllib.request
from collections import Counter

random.seed(42)  # 레포 관례; BPE 자체는 완전히 결정적임


# === CONSTANTS ===

NUM_MERGES = 256  # 최종 어휘 = 256 바이트 토큰 + 256 병합 = 512 토큰.
# 참고: 프로덕션 토크나이저(GPT-2, GPT-4)는 수백 기가바이트에서 학습된 50K+ 병합을 사용함.
# 18KB에 대한 256 병합은 장난감 수준이지만, 알고리즘은 동일함.

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"


# === DATA LOADING ===

def load_data(url: str, filename: str) -> bytes:
    """캐시되지 않은 경우 데이터셋을 다운로드하고, 원시 바이트를 반환함."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "rb") as f:
        return f.read()


# === BPE TRAINING ===

def get_pair_counts(token_ids: list[int]) -> Counter:
    """모든 인접 토큰 쌍의 빈도를 셈.

    시퀀스 s = [s_0, s_1, ..., s_n]에서 모든 (s_i, s_{i+1}) 쌍을 셈.
    예시: [a, b, c, b, c] -> {(a,b): 1, (b,c): 2, (c,b): 1}.
    BPE가 다음에 무엇을 병합할지 결정하는 데 사용하는 핵심 통계임.
    """
    # zip(ids, ids[1:])는 각 원소를 오른쪽 이웃과 짝지음 -- O(n).
    return Counter(zip(token_ids, token_ids[1:]))


def apply_merge(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """왼쪽에서 오른쪽으로 한 번 스캔하며 `pair`의 모든 출현을 `new_id`로 교체함.

    겹치는 쌍은 왼쪽에서 오른쪽으로 해소됨: [a, a, a]에서 (a,a)를 병합하면
    [new, a]가 됨, [a, new]가 아님. 이것이 표준 BPE 관례이며, 쌍 겹침 패턴에
    관계없이 병합 연산이 결정적임을 보장함.
    """
    # 참고: 이 O(n) 스캔은 병합당 한 번 실행되어, M번 병합에 총 O(n * M) 학습
    # 비용이 듦. 프로덕션 구현(SentencePiece, tiktoken)은 총 O(n log n)을 위해
    # 우선순위 큐를 쓰지만, 출력은 동일함.
    merged = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2  # 쌍의 두 토큰을 소비함
        else:
            merged.append(token_ids[i])
            i += 1
    return merged


def train_bpe(
    token_ids: list[int], num_merges: int
) -> list[tuple[tuple[int, int], int]]:
    """가장 빈번한 인접 쌍을 탐욕적으로 병합하여 BPE 병합 규칙을 학습함.

    각 병합은 코퍼스에서 가장 중복된 단일 쌍을 흡수함 -- 언어적 규칙 없이
    형태소 단위("an" + "a", "el" + "la")를 자연스럽게 발견하는 탐욕적 압축 단계임.
    병합 테이블은 우선순위로 정렬됨: 병합 0이 원래 코퍼스에서 가장 빈번했고,
    병합 1이 병합 0 이후에 가장 빈번했고, 이런 식으로 계속됨. 이 순서가 인코딩에 중요함.

    반환: (pair, new_id) 튜플의 정렬된 리스트, new_id = 256 + merge_index.
    """
    ids = list(token_ids)  # 복사본에서 작업함
    merges: list[tuple[tuple[int, int], int]] = []

    for i in range(num_merges):
        counts = get_pair_counts(ids)
        if not counts:
            # 전체 코퍼스가 단일 토큰으로 축소됨 (또는 비어 있음). 실제로는 드물지만,
            # 올바르게 처리해야 함: 더 이상 쌍이 없으면 더 이상 병합할 수 없음.
            break

        # 가장 높은 빈도를 가진 쌍이 다음에 병합됨.
        pair = max(counts, key=counts.get)  # type: ignore[arg-type]
        new_id = 256 + i  # 바이트 ID 0-255는 예약됨; 병합은 256부터 시작함

        ids = apply_merge(ids, pair, new_id)
        merges.append((pair, new_id))

        if (i + 1) % 32 == 0 or i == 0:
            a, b = pair
            print(
                f"  merge {i + 1:>3}/{num_merges}: "
                f"({a:>3}, {b:>3}) -> {new_id:>3}  "
                f"freq={counts[pair]:>5}  corpus_len={len(ids)}"
            )

    return merges


# === ENCODING & DECODING ===

def build_vocab(merges: list[tuple[tuple[int, int], int]]) -> dict[int, bytes]:
    """토큰 ID -> bytes 조회 테이블을 구축함.

    기본 어휘: 각 바이트 값을 단일 바이트 문자열로 매핑하는 256개 항목.
    각 병합은 테이블을 확장함: vocab[new_id] = vocab[a] + vocab[b].
    이 재귀적 확장 덕분에 디코딩은 단순한 테이블 조회로 가능함 -- 병합을 재생할
    필요가 없고, 왕복 정확성이 구조적으로 보장됨.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges:
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


def encode(text: str, merges: list[tuple[tuple[int, int], int]]) -> list[int]:
    """병합을 우선순위 순서로 재생하여 문자열을 BPE 토큰 ID로 인코딩함.

    중요: 병합은 학습된 순서(우선순위 순서)로 적용되며, 새 텍스트에서 빈도를 다시
    세는 것이 아님. 우선순위 순서가 결정적 토큰화를 보장함 -- 같은 문자열은 항상
    같은 토큰 시퀀스를 생성하며, 토크나이저가 어떤 다른 텍스트에서 학습되었든
    상관없음. 빈도를 다시 세면 출력이 입력 배치에 의존하게 되어, 토큰화가 입력
    문자열의 순수 함수라는 계약을 깨뜨림.
    """
    # 참고: 이 O(n * M) 단순 인코딩은 매 병합을 전체 시퀀스에 대해 확인함.
    # 프로덕션 토크나이저(tiktoken, HuggingFace)는 O(n) 인코딩을 위해 trie 구조를
    # 쓰지만, 동일한 출력을 생성함.
    token_ids = list(text.encode("utf-8"))
    for pair, new_id in merges:
        token_ids = apply_merge(token_ids, pair, new_id)
    return token_ids


def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """바이트 조회와 UTF-8 디코딩으로 토큰 ID를 문자열로 되돌림.

    모든 토큰은 vocab 테이블을 통해 확정된 바이트 시퀀스에 매핑되므로,
    유효한 UTF-8 입력에 대해 decode(encode(text)) == text가 보장됨.
    디코딩은 설계상 매우 단순함 -- 모든 복잡성은 인코딩에 있음.
    """
    raw_bytes = b"".join(vocab[tid] for tid in token_ids)
    return raw_bytes.decode("utf-8")


# === INFERENCE DEMO ===

if __name__ == "__main__":
    # -- 데이터 로드 및 준비 --
    raw = load_data(DATA_URL, DATA_FILE)
    corpus_ids = list(raw)

    # 원시 바이트에서 시작한다는 것은 모든 가능한 입력을 표현할 수 있다는 뜻임 --
    # "알 수 없는 토큰" 문제가 없음. 이것이 바이트 수준 BPE의 핵심 통찰임:
    # 기본 어휘가 모든 유니코드를 (UTF-8 바이트 시퀀스를 통해) 커버하므로
    # 모든 문자 체계에 대한 문자 수준 어휘가 필요 없음.
    print(f"Corpus: {len(raw):,} bytes, base vocab: 256 byte tokens")
    print(f"Training {NUM_MERGES} merges (final vocab: {256 + NUM_MERGES} tokens)\n")

    # -- 학습 --
    print("Training BPE...")
    merges = train_bpe(corpus_ids, NUM_MERGES)
    vocab = build_vocab(merges)
    print(f"\nTraining complete: {len(merges)} merges learned\n")

    # -- 왕복 테스트 --
    # 다양한 입력에서 encode-decode 항등성을 검증함: 일반 이름, 희귀 이름,
    # 하이픈, 아포스트로피, 빈 문자열, 단일 문자.
    test_strings = ["Emma", "Xiomara", "Mary-Jane", "O'Brien", "", "Z"]
    print("Round-trip tests:")
    all_pass = True
    for s in test_strings:
        encoded = encode(s, merges)
        decoded = decode(encoded, vocab)
        status = "PASS" if decoded == s else "FAIL"
        if status == "FAIL":
            all_pass = False
        display = f'"{s}"' if s else '""'
        print(f"  [{status}] {display:<14} -> {len(encoded):>2} tokens -> {decoded!r}")
    print()

    # -- 압축률 --
    # compression_ratio = len(original_bytes) / len(bpe_tokens)
    # 각 BPE 토큰이 평균적으로 `ratio`바이트를 나타냄. 높을수록 좋음 --
    # 토크나이저가 더 압축 가능한 구조를 발견했다는 뜻임.
    corpus_text = raw.decode("utf-8")
    corpus_encoded = encode(corpus_text, merges)
    ratio = len(raw) / len(corpus_encoded)
    print(
        f"Compression: {len(raw):,} bytes -> {len(corpus_encoded):,} tokens "
        f"(ratio: {ratio:.2f}x)\n"
    )

    # -- 상위 20개 병합 --
    print("Top 20 merges (earliest = highest priority):")
    for i, ((a, b), new_id) in enumerate(merges[:20]):
        a_str = vocab[a].decode("utf-8", errors="replace")
        b_str = vocab[b].decode("utf-8", errors="replace")
        merged_str = vocab[new_id].decode("utf-8", errors="replace")
        print(f"  {i + 1:>2}. {a_str!r:>6} + {b_str!r:<6} -> {merged_str!r}")
    print()

    # -- 토큰화 예시 --
    example = "Elizabeth"
    example_tokens = encode(example, merges)
    pieces = [vocab[tid].decode("utf-8", errors="replace") for tid in example_tokens]
    print(f'Tokenization example: "{example}"')
    print(f"  Bytes:  {list(example.encode('utf-8'))}")
    print(f"  Tokens: {example_tokens}")
    print(f"  Pieces: {pieces}")
