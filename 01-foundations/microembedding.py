"""
의미가 기하학이 되는 방법 — character n-gram과 contrastive loss만으로
거리가 유사도와 같아지는 벡터를 학습함.
"""
# Contrastive embedding 학습(InfoNCE loss)은 희소한 고차원 character n-gram 특징을
# cosine similarity가 의미적 유사도를 반영하는 밀집 저차원 벡터로 변환함.
# SimCLR과 sentence-transformers에서 영감을 받았지만, 딥 네트워크 없이
# 선형 projection으로 단순화함.

from __future__ import annotations

import math
import os
import random
import urllib.request
from collections import Counter

random.seed(42)


# === CONSTANTS ===

EMBEDDING_DIM = 32  # 목표 embedding 차원 (희소 n-gram → 밀집 벡터)
LEARNING_RATE = 0.05  # SGD learning rate (Adam은 여기서 큰 이점 없이 오버헤드만 추가함)
TEMPERATURE = 0.1  # InfoNCE temperature: 낮을수록 유사도 분포가 더 날카로움
NUM_EPOCHS = 30  # positive/random 쌍 간의 명확한 분리를 볼 수 있을 만큼 충분함
BATCH_SIZE = 64
MAX_VOCAB = 500  # 속도를 위해 n-gram 어휘를 가장 빈번한 항목으로 제한
TRAIN_SIZE = 5000  # 학습용 이름 서브셋 (전체 32K는 데모에 너무 느림)

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# 참고: 프로덕션 embedding 모델(sentence-transformers, CLIP)은 12+ 레이어의 딥
# transformer 인코더를 사용함. 이 선형 projection은 아키텍처의 복잡성 없이
# 핵심 contrastive learning 메커니즘을 보여줌.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """데이터셋이 캐시되어 있지 않으면 다운로드하고, 이름 리스트를 반환함."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]


# === FEATURE EXTRACTION ===

def extract_ngrams(text: str) -> list[str]:
    """텍스트에서 character bigram과 trigram을 추출함.

    n-gram을 쓰는 이유: 개별 문자보다 로컬 발음 패턴을 더 잘 포착함.
    "anna"와 "anne"는 bigram "an", "nn"과 trigram "ann"을 공유해서,
    한 글자만 달라도 n-gram 벡터의 겹침이 높음. 이것이 n-gram이
    발음 유사도에 민감한 이유임.
    """
    # 시작/끝 패턴을 포착하기 위해 경계 마커로 패딩
    padded = f"^{text}$"
    bigrams = [padded[i:i+2] for i in range(len(padded) - 1)]
    trigrams = [padded[i:i+3] for i in range(len(padded) - 2)]
    return bigrams + trigrams


def build_ngram_vocab(names: list[str], max_vocab: int) -> dict[str, int]:
    """가장 빈번한 n-gram을 인덱스에 매핑하는 어휘를 구축함.

    어휘 제한의 두 가지 목적: (1) 성능 — gradient 루프가
    O(non_zero_ngrams * embedding_dim)이고, (2) 품질 — 한두 번만 본
    희귀 n-gram은 유용한 패턴 학습 없이 노이즈만 추가함.
    """
    counts: Counter[str] = Counter()
    for name in names:
        counts.update(extract_ngrams(name))

    # 가장 빈번한 상위 max_vocab개 n-gram을 유지
    most_common = counts.most_common(max_vocab)
    return {ngram: idx for idx, (ngram, _) in enumerate(most_common)}


def encode_ngrams_sparse(text: str, vocab: dict[str, int]) -> dict[int, float]:
    """텍스트를 희소 n-gram 카운트 dict(인덱스 → 카운트)로 변환함.

    0이 아닌 항목만 반환함. 성능에 중요함: 이름은 500개 어휘 중 ~10-15개
    n-gram만 가지므로, 희소 표현이 gradient와 인코더 루프에서 97% 계산을 건너뜀.
    """
    sparse: dict[int, float] = {}
    for ngram in extract_ngrams(text):
        if ngram in vocab:
            idx = vocab[ngram]
            sparse[idx] = sparse.get(idx, 0.0) + 1.0
    return sparse


# === AUGMENTATION ===

def augment(name: str) -> str:
    """랜덤 문자 삭제 또는 교환으로 positive pair를 생성함.

    augmentation을 쓰는 이유: 인코더가 작은 변화에 대한 불변성을 학습하게 강제함.
    "anna"와 "ana"가 비슷한 embedding에 매핑되면, 모델이 문자 삭제가 정체성을
    보존한다는 걸 학습한 것임 — 유사한 입력이 유사한 표현을 가져야 한다는
    contrastive learning 원리임.
    """
    if len(name) <= 2:
        return name  # 안전하게 augment하기엔 너무 짧음

    if random.random() < 0.5:
        # 랜덤 문자 하나 삭제
        idx = random.randint(0, len(name) - 1)
        return name[:idx] + name[idx + 1:]
    else:
        # 인접한 두 문자 교환
        idx = random.randint(0, len(name) - 2)
        chars = list(name)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)


# === ENCODER ===

def l2_normalize(vec: list[float]) -> list[float]:
    """벡터를 단위 길이로 정규화함.

    L2 정규화를 쓰는 이유: embedding을 단위 초구면에 제약함. 정규화 후
    cosine similarity = dot product가 되어 수학이 단순해지고 embedding 공간이
    등방적(모든 방향의 분산이 동일)이 됨. contrastive learning(SimCLR, CLIP)의
    표준 관행임.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-10:
        return vec
    return [x / norm for x in vec]


def encode_sparse_raw(
    sparse_ngrams: dict[int, float], W: list[list[float]]
) -> list[float]:
    """희소 n-gram 특징을 정규화 없이 embedding 공간에 투영함.

    수식: z = W @ x (raw, 비정규화 embedding)
    학습 시 정규화를 통한 역전파가 필요할 때 사용됨.
    """
    embedding = [0.0] * len(W)
    for i in range(len(W)):
        total = 0.0
        for j, count in sparse_ngrams.items():
            total += W[i][j] * count
        embedding[i] = total
    return embedding


def encode_sparse(
    sparse_ngrams: dict[int, float], W: list[list[float]]
) -> list[float]:
    """희소 n-gram 특징을 embedding 공간에 투영하고 정규화함.

    수식: emb = normalize(W @ x)
    희소 버전: 전체 500개 어휘 대신 x에서 0이 아닌 항목(10-15개 n-gram)만
    합산함.
    """
    return l2_normalize(encode_sparse_raw(sparse_ngrams, W))


def grad_through_norm(
    raw_emb: list[float], grad_normalized: list[float]
) -> list[float]:
    """L2 정규화를 통해 gradient를 역전파함.

    z = raw_emb이고 e = z/||z|| (정규화된 embedding)이면:
        d(L)/d(z_i) = (g_i - e_i * dot(g, e)) / ||z||

    정규화 Jacobian이 gradient의 반경 방향 성분을 제거하고 단위 구면의
    접선 방향만 남김. 이 투영이 없으면 gradient가 모든 embedding을 같은
    반경 방향으로 밀어서 "representation collapse"를 일으킴 — contrastive
    learning에서 가장 흔한 실패 모드임.
    """
    norm = math.sqrt(sum(x * x for x in raw_emb))
    if norm < 1e-10:
        return list(grad_normalized)
    e = [x / norm for x in raw_emb]
    g_dot_e = sum(g * ei for g, ei in zip(grad_normalized, e))
    return [(g - ei * g_dot_e) / norm for g, ei in zip(grad_normalized, e)]


# === SIMILARITY ===

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 L2 정규화된 벡터 간의 cosine similarity를 계산함.

    벡터가 L2 정규화되어 있으므로 cosine similarity = dot product임.
    범위: [-1, 1] 여기서 1 = 동일 방향, -1 = 반대, 0 = 직교.
    """
    return sum(a[i] * b[i] for i in range(len(a)))


# === INFONCE LOSS ===

def infonce_loss_and_grads(
    anchor_embs: list[list[float]],
    positive_embs: list[list[float]],
    temperature: float,
) -> tuple[float, list[list[float]], list[list[float]]]:
    """InfoNCE (NT-Xent) loss와 embedding 공간 gradient를 계산함.

    배치의 각 (anchor, positive) 쌍에 대해, loss는 positive와의 높은 유사도와
    모든 negative(배치 내 다른 샘플)와의 낮은 유사도를 장려함.

    수식 (anchor i에 대해):
        sim_pos = cos(anchor_i, positive_i) / tau
        sim_neg_j = cos(anchor_i, anchor_j) / tau   for j != i
        loss_i = -log(exp(sim_pos) / (exp(sim_pos) + sum_j exp(sim_neg_j)))

    temperature를 쓰는 이유: 유사도 분포의 날카로움을 제어함. 낮은 tau(예: 0.1)는
    loss가 hard negative에 집중하게 함. tau=0.1은 SimCLR의 표준임.

    반환: (avg_loss, anchor_grads, positive_grads)
    """
    bs = len(anchor_embs)
    total_loss = 0.0

    anchor_grads = [[0.0] * EMBEDDING_DIM for _ in range(bs)]
    positive_grads = [[0.0] * EMBEDDING_DIM for _ in range(bs)]

    for i in range(bs):
        # positive pair와의 유사도
        sim_pos = cosine_similarity(anchor_embs[i], positive_embs[i]) / temperature

        # 모든 negative(배치 내 다른 anchor)와의 유사도
        sim_negs = []
        for j in range(bs):
            if j != i:
                sim_negs.append(
                    cosine_similarity(anchor_embs[i], anchor_embs[j]) / temperature
                )

        # 수치 안정성을 위한 log-sum-exp 트릭 (exp 전에 max를 뺌)
        max_sim = max([sim_pos] + sim_negs)
        exp_pos = math.exp(sim_pos - max_sim)
        exp_negs = [math.exp(s - max_sim) for s in sim_negs]
        denom = exp_pos + sum(exp_negs)

        # Loss: positive pair의 softmax 확률의 -log
        total_loss += -math.log(max(exp_pos / denom, 1e-10))

        # anchor embedding에 대한 loss의 gradient:
        # d(loss)/d(anchor_i) = (1/tau) * (sum_j p_j * anchor_j - positive_i)
        # 여기서 p_j = exp(sim_neg_j) / denom은 softmax 확률

        # Positive 기여: anchor를 positive 쪽으로 밀음
        p_pos = exp_pos / denom
        for d in range(EMBEDDING_DIM):
            anchor_grads[i][d] += (p_pos - 1.0) / temperature * positive_embs[i][d]
            positive_grads[i][d] += (p_pos - 1.0) / temperature * anchor_embs[i][d]

        # Negative 기여: anchor를 negative로부터 멀리 밀음
        neg_idx = 0
        for j in range(bs):
            if j == i:
                continue
            p_neg = exp_negs[neg_idx] / denom
            for d in range(EMBEDDING_DIM):
                anchor_grads[i][d] += p_neg / temperature * anchor_embs[j][d]
            neg_idx += 1

    return total_loss / bs, anchor_grads, positive_grads


# === TRAINING ===

def train(
    names: list[str],
    vocab: dict[str, int],
    W: list[list[float]],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """SGD로 embedding 모델을 학습함.

    참고: 프로덕션 시스템은 learning rate warmup이 있는 Adam을 사용함. SGD가
    여기서 충분한 이유는 모델이 단일 선형 레이어라서 레이어 간 gradient 스케일
    문제가 없기 때문임.
    """
    vocab_size = len(vocab)

    for epoch in range(num_epochs):
        epoch_names = names[:]
        random.shuffle(epoch_names)

        epoch_loss = 0.0
        num_batches = 0

        for batch_start in range(0, len(epoch_names), batch_size):
            batch = epoch_names[batch_start:batch_start + batch_size]
            if len(batch) < 2:
                continue

            # anchor와 positive를 인코딩 (희소 n-gram → 밀집 embedding)
            # raw(정규화 전)와 정규화된 embedding 모두 저장:
            # 역전파에서 정규화 Jacobian에 raw embedding이 필요함
            anchor_sparse = []
            positive_sparse = []
            anchor_raw = []
            positive_raw = []
            anchor_embs = []
            positive_embs = []

            for name in batch:
                a_sp = encode_ngrams_sparse(name, vocab)
                anchor_sparse.append(a_sp)
                a_raw = encode_sparse_raw(a_sp, W)
                anchor_raw.append(a_raw)
                anchor_embs.append(l2_normalize(a_raw))

                p_sp = encode_ngrams_sparse(augment(name), vocab)
                positive_sparse.append(p_sp)
                p_raw = encode_sparse_raw(p_sp, W)
                positive_raw.append(p_raw)
                positive_embs.append(l2_normalize(p_raw))

            # 정규화된 embedding에 대한 loss와 gradient를 계산
            loss, a_grads, p_grads = infonce_loss_and_grads(
                anchor_embs, positive_embs, TEMPERATURE
            )
            epoch_loss += loss
            num_batches += 1

            # 희소 연산으로 W에 gradient를 역전파함.
            # Chain rule: d(L)/d(W) = d(L)/d(emb_norm) * d(emb_norm)/d(emb_raw) * d(emb_raw)/d(W)
            # 정규화 Jacobian(중간 항)이 반경 방향 gradient 성분을 제거하여
            # representation collapse를 방지함.
            grad_W = [[0.0] * vocab_size for _ in range(EMBEDDING_DIM)]

            for b_idx in range(len(batch)):
                # 정규화 Jacobian을 통해 gradient를 변환
                a_grad_raw = grad_through_norm(anchor_raw[b_idx], a_grads[b_idx])
                p_grad_raw = grad_through_norm(positive_raw[b_idx], p_grads[b_idx])

                for j, count in anchor_sparse[b_idx].items():
                    for i in range(EMBEDDING_DIM):
                        grad_W[i][j] += a_grad_raw[i] * count

                for j, count in positive_sparse[b_idx].items():
                    for i in range(EMBEDDING_DIM):
                        grad_W[i][j] += p_grad_raw[i] * count

            # SGD 업데이트 (0이 아닌 gradient가 있는 항목만)
            scale = learning_rate / len(batch)
            for i in range(EMBEDDING_DIM):
                for j in range(vocab_size):
                    if grad_W[i][j] != 0.0:
                        W[i][j] -= scale * grad_W[i][j]

        avg_loss = epoch_loss / max(num_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3}/{num_epochs}  loss={avg_loss:.4f}")


# === INFERENCE ===

def find_nearest_neighbors(
    query: str,
    candidates: list[str],
    vocab: dict[str, int],
    W: list[list[float]],
    k: int = 5,
) -> list[tuple[str, float]]:
    """embedding 공간에서 cosine similarity로 k개의 최근접 이웃을 찾음."""
    q_emb = encode_sparse(encode_ngrams_sparse(query, vocab), W)

    similarities = []
    for candidate in candidates:
        if candidate == query:
            continue
        c_emb = encode_sparse(encode_ngrams_sparse(candidate, vocab), W)
        sim = cosine_similarity(q_emb, c_emb)
        similarities.append((candidate, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


# === MAIN ===

if __name__ == "__main__":
    # 데이터 로드
    all_names = load_data(DATA_URL, DATA_FILE)
    print(f"Loaded {len(all_names):,} names")

    # 속도를 위해 학습 서브셋을 사용하고, 최근접 이웃 검색에는 전체 이름을 유지함
    train_names = all_names[:TRAIN_SIZE]
    print(f"Training on {len(train_names):,} names\n")

    # 학습 세트에서 n-gram 어휘를 구축 (MAX_VOCAB로 제한)
    print("Building n-gram vocabulary...")
    vocab = build_ngram_vocab(train_names, MAX_VOCAB)
    print(f"Vocabulary: {len(vocab)} n-grams (top {MAX_VOCAB} most frequent)\n")

    # projection 행렬 W를 초기화: [embedding_dim x vocab_size]
    W = [
        [random.gauss(0, 0.01) for _ in range(len(vocab))]
        for _ in range(EMBEDDING_DIM)
    ]
    num_params = EMBEDDING_DIM * len(vocab)
    print(f"Model: linear projection ({EMBEDDING_DIM} x {len(vocab)} = {num_params:,} params)\n")

    # 학습
    print(f"Training (epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, temp={TEMPERATURE})...")
    train(train_names, vocab, W, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
    print()

    # === EVALUATION ===

    # Positive pair: 비슷하게 발음되는 이름 (높은 유사도를 가져야 함)
    positive_pairs = [
        ("anna", "anne"), ("john", "jon"), ("elizabeth", "elisabeth"),
        ("michael", "michelle"), ("alexander", "alexandra"),
    ]

    # Random pair: 다른 이름 (낮은 유사도를 가져야 함)
    random_pairs = [
        ("anna", "zachary"), ("john", "penelope"), ("elizabeth", "bob"),
        ("michael", "quinn"), ("alexander", "ivy"),
    ]

    print("Positive pairs (should be similar):")
    pos_sims = []
    for name1, name2 in positive_pairs:
        e1 = encode_sparse(encode_ngrams_sparse(name1, vocab), W)
        e2 = encode_sparse(encode_ngrams_sparse(name2, vocab), W)
        sim = cosine_similarity(e1, e2)
        pos_sims.append(sim)
        print(f"  {name1:<12} <-> {name2:<12}  sim={sim:>6.3f}")

    print("\nRandom pairs (should be dissimilar):")
    rand_sims = []
    for name1, name2 in random_pairs:
        e1 = encode_sparse(encode_ngrams_sparse(name1, vocab), W)
        e2 = encode_sparse(encode_ngrams_sparse(name2, vocab), W)
        sim = cosine_similarity(e1, e2)
        rand_sims.append(sim)
        print(f"  {name1:<12} <-> {name2:<12}  sim={sim:>6.3f}")

    avg_pos = sum(pos_sims) / len(pos_sims)
    avg_rand = sum(rand_sims) / len(rand_sims)
    print(f"\nAverage positive pair similarity: {avg_pos:.3f}")
    print(f"Average random pair similarity:   {avg_rand:.3f}")

    # 최근접 이웃 검색 데모
    # 더 흥미로운 결과를 위해 더 큰 풀에서 검색
    search_pool = all_names[:10000]
    query_names = ["anna", "john", "elizabeth", "michael"]
    print("\nNearest neighbor retrieval:")
    for query in query_names:
        neighbors = find_nearest_neighbors(query, search_pool, vocab, W, k=5)
        neighbor_str = ", ".join(f"{n} ({s:.2f})" for n, s in neighbors)
        print(f"  {query:<12} -> {neighbor_str}")
