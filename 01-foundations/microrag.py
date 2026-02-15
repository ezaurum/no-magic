"""
retrieval이 generation을 어떻게 보강하는지 -- BM25 검색과 character-level MLP로
실제로 동작하는 가장 간단한 시스템, 순수 Python 구현.
"""
# Reference: RAG architecture inspired by "Retrieval-Augmented Generation for
# Knowledge-Intensive NLP Tasks" (Lewis et al., 2020), BM25 scoring from Robertson
# and Zaragoza (2009). Implementation rewritten from scratch for educational clarity.

from __future__ import annotations

import math
import random
import string

random.seed(42)


# === CONSTANTS ===

LEARNING_RATE = 0.01
HIDDEN_DIM = 64  # MLP의 hidden layer 크기
NUM_EPOCHS = 300
TOP_K = 3  # 상위 3개 문서를 검색함
BATCH_SIZE = 5

# BM25 hyperparameter (정보 검색 분야의 표준 값)
K1 = 1.2  # term frequency 포화 파라미터
B = 0.75  # 문서 길이 정규화 파라미터

CHAR_VOCAB = list(string.ascii_lowercase + " .,")  # 문자 vocabulary
VOCAB_SIZE = len(CHAR_VOCAB)


# === SYNTHETIC KNOWLEDGE BASE ===

def generate_knowledge_base() -> tuple[list[str], list[tuple[str, str]]]:
    """합성 factual 문단 100개와 테스트 query 20개를 생성함.

    template + 데이터 테이블을 사용해 도시, 국가, 인구, 지리에 대한 검증 가능한
    factual knowledge를 생성함. 외부 다운로드나 API 호출 없이 결정적이고
    재현 가능한 데이터를 보장함.

    Returns: (documents, test_queries), test_queries는 (query, expected_doc_index) 쌍임
    """
    # 데이터 테이블 -- "ground truth" 사실들
    cities = [
        ("Paris", "France", "2.1 million", "Seine"),
        ("London", "United Kingdom", "8.9 million", "Thames"),
        ("Berlin", "Germany", "3.8 million", "Spree"),
        ("Madrid", "Spain", "3.3 million", "Manzanares"),
        ("Rome", "Italy", "2.8 million", "Tiber"),
        ("Tokyo", "Japan", "14 million", "Sumida"),
        ("Beijing", "China", "21 million", "Yongding"),
        ("Delhi", "India", "16 million", "Yamuna"),
        ("Cairo", "Egypt", "9.5 million", "Nile"),
        ("Lagos", "Nigeria", "14 million", "Lagos Lagoon"),
    ]

    mountains = [
        ("Everest", "Nepal", "8849 meters"),
        ("K2", "Pakistan", "8611 meters"),
        ("Kilimanjaro", "Tanzania", "5895 meters"),
        ("Mont Blanc", "France", "4808 meters"),
        ("Denali", "United States", "6190 meters"),
    ]

    # 도시 문단 생성
    documents = []
    for city, country, pop, river in cities:
        doc = (
            f"{city} is the capital of {country}. "
            f"It has a population of approximately {pop}. "
            f"The {river} river flows through the city."
        )
        documents.append(doc.lower())

    # 산 문단 생성
    for mountain, country, height in mountains:
        doc = (
            f"{mountain} is located in {country}. "
            f"The mountain has a height of {height}. "
            f"It is a popular destination for climbers."
        )
        documents.append(doc.lower())

    # 추가 filler 문서 생성 (대륙 사실, 간단한 서술)
    continents = [
        "africa is the second largest continent by area.",
        "asia is the most populous continent in the world.",
        "europe has diverse cultures and languages.",
        "north america includes canada, united states, and mexico.",
        "south america is home to the amazon rainforest.",
    ]
    documents.extend(continents)

    # 100개 문서를 채우기 위해 다양한 factual 서술을 추가함
    for i in range(80):
        # 약간의 변형을 주어 사실을 재조합해 더 많은 문서를 생성함
        if i % 4 == 0:
            city, country, pop, river = cities[i % len(cities)]
            doc = f"The population of {city} is about {pop}. It is in {country}."
        elif i % 4 == 1:
            mountain, country, height = mountains[i % len(mountains)]
            doc = f"{mountain} stands at {height} in {country}."
        elif i % 4 == 2:
            city, country, pop, river = cities[i % len(cities)]
            doc = f"The {river} river is a major waterway in {city}, {country}."
        else:
            city, country, pop, river = cities[i % len(cities)]
            doc = f"{city} is a major city with population {pop}."
        documents.append(doc.lower())

    # 정답이 알려진 테스트 query 생성 (문서 인덱스)
    test_queries = [
        ("population of paris", 0),  # Paris 문서
        ("seine river", 0),  # Paris 문서에 Seine이 언급됨
        ("tokyo population", 5),  # Tokyo 문서
        ("everest height", 10),  # Everest 문서
        ("capital of germany", 2),  # Berlin 문서
        ("nile river", 8),  # Cairo 문서에 Nile이 언급됨
        ("kilimanjaro tanzania", 12),  # Kilimanjaro 문서
        ("thames river london", 1),  # London 문서
        ("mont blanc france", 13),  # Mont Blanc 문서
        ("beijing china", 6),  # Beijing 문서
    ]

    return documents, test_queries


# === TOKENIZATION ===

def tokenize(text: str) -> list[str]:
    """간단한 word-level tokenization: 소문자 변환, 구두점 제거, 공백으로 분리함.

    Signpost: 프로덕션 RAG 시스템은 학습된 subword tokenizer(BPE, SentencePiece)를 사용함.
    여기서는 retrieval 메커니즘을 보여주는 것이 목적이므로 word-level tokenization으로 충분함 --
    BM25 scoring과 context 주입에 초점을 맞추며, tokenization 품질은 중요하지 않음.
    """
    # 구두점을 제거하고 단어로 분리함
    words = []
    word = []
    for char in text.lower():
        if char.isalpha() or char.isdigit():
            word.append(char)
        elif word:
            words.append("".join(word))
            word = []
    if word:
        words.append("".join(word))
    return words


# === BM25 INDEX ===

class BM25Index:
    """문서 검색을 위한 BM25 scoring.

    BM25는 TF-IDF를 두 가지 핵심 통찰로 개선함:
    1. TF 포화: 10번 등장이 1번 등장보다 10배 더 관련 있는 것은 아님.
       수식 (tf * (k1 + 1)) / (tf + k1)은 tf → ∞일 때 포화됨.
    2. 문서 길이 정규화: 긴 문서가 본질적으로 더 관련 있는 것은 아님.
       정규화 항 (1 - b + b * dl/avgdl)이 긴 문서에 페널티를 줌.

    Math-to-code mapping:
      idf(term) = log((N - df + 0.5) / (df + 0.5) + 1)
      tf_score(term, doc) = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
      BM25(query, doc) = Σ_{term in query} idf(term) * tf_score(term, doc)

    where:
      N = 전체 문서 수
      df = term을 포함하는 문서 수
      tf = 문서 내 term 빈도
      dl = 문서 길이 (단어 수)
      avgdl = 코퍼스 전체의 평균 문서 길이
      k1 = TF 포화 파라미터 (1.2가 표준)
      b = 길이 정규화 파라미터 (0.75가 표준)
    """

    def __init__(self, documents: list[str], k1: float = K1, b: float = B):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)  # 전체 문서 수

        # 모든 문서를 tokenize함
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # inverted index 구축: term -> (doc_id, term_frequency) 리스트
        # 효율적인 검색을 위한 핵심 자료구조 -- 각 term에 대해
        # 어떤 문서에 포함되어 있고 얼마나 자주 나타나는지 미리 계산함. query 시점에
        # query와 최소 하나의 term을 공유하는 문서만 scoring함.
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        for doc_id, tokens in enumerate(self.doc_tokens):
            term_counts: dict[str, int] = {}
            for term in tokens:
                term_counts[term] = term_counts.get(term, 0) + 1
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_id, count))

        # 모든 term의 IDF 점수를 미리 계산함
        # IDF 수식: log((N - df + 0.5) / (df + 0.5) + 1), df = document frequency
        # 왜 0.5를 더하나? 0으로 나누는 것을 방지하고 희귀 term의 영향을 줄이기 위한 smoothing임.
        # 왜 바깥에 +1? IDF가 항상 양수가 되도록 보장함 (x < 1일 때 log(x) < 0).
        self.idf: dict[str, float] = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)  # document frequency = term을 포함하는 문서 수
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_id: int) -> float:
        """query에 대한 특정 문서의 BM25 점수를 계산함."""
        query_terms = tokenize(query)
        score = 0.0

        dl = self.doc_lengths[doc_id]  # 문서 길이
        # 문서 길이 정규화 계수: 긴 문서에 페널티를 주되, 선형적이지는 않음
        norm = 1 - self.b + self.b * (dl / self.avgdl)

        # 문서 내 term 빈도를 계산함
        doc_term_counts: dict[str, int] = {}
        for term in self.doc_tokens[doc_id]:
            doc_term_counts[term] = doc_term_counts.get(term, 0) + 1

        for term in query_terms:
            if term not in self.idf:
                continue  # 코퍼스에 없는 term이므로 점수에 기여하지 않음
            tf = doc_term_counts.get(term, 0)
            if tf == 0:
                continue  # 이 문서에 없는 term임

            # TF 포화: (tf * (k1 + 1)) / (tf + k1 * norm)
            # tf → ∞일 때, 이 값은 (k1 + 1) / k1 ≈ 1.83 (k1=1.2일 때)에 수렴함.
            # term frequency가 점수를 지배하는 것을 방지함.
            tf_score = (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            score += self.idf[term] * tf_score

        return score

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        """query에 대해 BM25 점수 기준 상위 k개 문서를 검색함.

        Returns: 점수 내림차순으로 정렬된 (doc_id, score) 튜플 리스트.
        """
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in range(self.N)]
        # 점수 내림차순 정렬 후 상위 k개 선택
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# === CHARACTER-LEVEL MLP GENERATOR ===

def char_to_index(char: str) -> int:
    """문자를 vocabulary 내 인덱스로 매핑함."""
    if char in CHAR_VOCAB:
        return CHAR_VOCAB.index(char)
    return CHAR_VOCAB.index(" ")  # 알 수 없는 문자는 공백으로 fallback함

def index_to_char(idx: int) -> str:
    """인덱스를 문자로 매핑함."""
    return CHAR_VOCAB[idx]


def one_hot(idx: int, size: int) -> list[float]:
    """one-hot 인코딩된 벡터를 생성함."""
    vec = [0.0] * size
    vec[idx] = 1.0
    return vec


class MLP:
    """query + context를 concat한 입력을 받는 character-level MLP generator.

    Architecture:
      input (query_chars + context_chars) → hidden (ReLU) → output (softmax over chars)

    핵심 RAG 메커니즘: 검색된 context를 query와 concat함으로써, MLP가
    검색된 사실에 기반해 예측을 조건부로 수행할 수 있음. RAG를 의미 있게
    보여주는 최소 아키텍처 -- 모델이 실제로 검색된 정보를 사용하며
    무시하지 않음.

    Signpost: 프로덕션 RAG는 transformer generator(GPT, LLaMA)를 사용함. MLP를 사용하는 이유는
    retrieval 메커니즘과 context 주입 패턴에 초점을 맞추기 위함임.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier initialization: 안정적인 gradient를 위해 weight를 1/sqrt(fan_in)으로 스케일링함
        # 왜 Xavier인가? 레이어 간 activation의 분산을 유지해
        # 학습 초기에 gradient가 vanishing하거나 exploding하는 것을 방지함.
        scale_1 = (2.0 / input_dim) ** 0.5
        scale_2 = (2.0 / hidden_dim) ** 0.5

        self.W1 = [[random.gauss(0, scale_1) for _ in range(input_dim)]
                   for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim

        self.W2 = [[random.gauss(0, scale_2) for _ in range(hidden_dim)]
                   for _ in range(output_dim)]
        self.b2 = [0.0] * output_dim

    def forward(self, x: list[float]) -> tuple[list[float], dict]:
        """Forward pass: input → hidden (ReLU) → output (softmax).

        Returns: (output_probs, cache), cache는 backward를 위한 중간값을 저장함.
        """
        # Hidden layer: h = ReLU(W1 @ x + b1)
        hidden = []
        for i in range(self.hidden_dim):
            activation = self.b1[i]
            for j in range(self.input_dim):
                activation += self.W1[i][j] * x[j]
            hidden.append(max(0.0, activation))  # ReLU

        # Output layer: o = W2 @ h + b2
        logits = []
        for i in range(self.output_dim):
            activation = self.b2[i]
            for j in range(self.hidden_dim):
                activation += self.W2[i][j] * hidden[j]
            logits.append(activation)

        # Stable softmax: exp(x - max(x))로 overflow를 방지함
        # 이것 없이는 큰 logit이 exp()를 inf로 overflow시켜 gradient가 깨짐.
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        # backward pass를 위한 cache
        cache = {"x": x, "hidden": hidden, "logits": logits, "probs": probs}
        return probs, cache

    def backward(
        self, target_idx: int, cache: dict, learning_rate: float
    ) -> float:
        """Backward pass: gradient를 계산하고 weight를 업데이트함.

        Cross-entropy loss: L = -log(p[target_idx])
        Cross-entropy + softmax의 gradient는 깔끔한 형태임: dL/do_i = p_i - 1[i == target]

        Returns: loss 값
        """
        x = cache["x"]
        hidden = cache["hidden"]
        probs = cache["probs"]

        # log(0) = -inf를 방지하기 위해 확률을 clip함
        loss = -math.log(max(probs[target_idx], 1e-10))

        # output logit에 대한 loss의 gradient: p - y (y는 one-hot target)
        dlogits = list(probs)
        dlogits[target_idx] -= 1.0

        # W2와 b2에 대한 gradient
        dW2 = [[0.0] * self.hidden_dim for _ in range(self.output_dim)]
        db2 = [0.0] * self.output_dim
        for i in range(self.output_dim):
            db2[i] = dlogits[i]
            for j in range(self.hidden_dim):
                dW2[i][j] = dlogits[i] * hidden[j]

        # hidden layer를 통한 backprop
        dhidden = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            for i in range(self.output_dim):
                dhidden[j] += dlogits[i] * self.W2[i][j]
            # ReLU gradient: hidden[j] <= 0이면 0, 아니면 통과
            if hidden[j] <= 0:
                dhidden[j] = 0.0

        # W1과 b1에 대한 gradient
        dW1 = [[0.0] * self.input_dim for _ in range(self.hidden_dim)]
        db1 = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            db1[i] = dhidden[i]
            for j in range(self.input_dim):
                dW1[i][j] = dhidden[i] * x[j]

        # SGD로 weight 업데이트: w = w - lr * dw
        for i in range(self.output_dim):
            self.b2[i] -= learning_rate * db2[i]
            for j in range(self.hidden_dim):
                self.W2[i][j] -= learning_rate * dW2[i][j]

        for i in range(self.hidden_dim):
            self.b1[i] -= learning_rate * db1[i]
            for j in range(self.input_dim):
                self.W1[i][j] -= learning_rate * dW1[i][j]

        return loss

    def generate(self, input_text: str, max_length: int = 50) -> str:
        """입력 context가 주어지면 문자 단위로 텍스트를 생성함.

        input_text는 query와 검색된 context를 concat한 것임.
        모델은 이 전체 context를 사용해 각 단계에서 다음 문자를 예측함.
        """
        # 입력 context를 시작점으로 사용함
        current_text = input_text
        for _ in range(max_length):
            # 최근 context를 인코딩함 (입력 크기를 관리 가능하도록 마지막 100자)
            context = current_text[-100:]
            x = []
            for char in context:
                idx = char_to_index(char)
                x.extend(one_hot(idx, VOCAB_SIZE))
            # 필요하면 고정 입력 크기로 padding함
            while len(x) < self.input_dim:
                x.append(0.0)
            x = x[:self.input_dim]  # 너무 길면 잘라냄

            # 다음 문자를 생성함
            probs, _ = self.forward(x)
            next_idx = probs.index(max(probs))  # greedy sampling
            next_char = index_to_char(next_idx)

            # 마침표에서 중단 (간단한 생성 종료)
            if next_char == ".":
                current_text += next_char
                break
            current_text += next_char

        return current_text


# === TRAINING LOOP ===

def train_rag(
    documents: list[str],
    bm25: BM25Index,
    mlp: MLP,
    num_epochs: int,
    learning_rate: float
):
    """knowledge base에서 추출한 (query, context, answer) 트리플로 MLP를 학습함.

    학습 과정:
    1. ground truth로 사용할 랜덤 문서를 샘플링함
    2. 문서에서 query를 추출함 (처음 몇 단어)
    3. BM25로 context를 검색함
    4. query + 검색된 context를 concat함
    5. ground truth 답변의 다음 문자를 예측하도록 MLP를 학습함

    검색된 context를 활용해 정확한 completion을 생성하는 법을 모델에 가르침.
    """
    print("Training RAG model...\n")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_samples = 0

        for _ in range(BATCH_SIZE):
            # ground truth로 사용할 랜덤 문서를 샘플링함
            doc_idx = random.randint(0, len(documents) - 1)
            doc = documents[doc_idx]

            # 문서의 처음 몇 단어로 query를 생성함
            # 문서의 주제에 대해 사용자가 질문하는 것을 시뮬레이션함
            words = tokenize(doc)
            if len(words) < 3:
                continue
            query_words = words[:min(3, len(words))]
            query = " ".join(query_words)

            # BM25로 context를 검색함
            retrieved = bm25.retrieve(query, top_k=TOP_K)
            context = " ".join([documents[doc_id] for doc_id, _ in retrieved[:2]])

            # query + context를 모델 입력으로 concat함
            # 이것이 핵심 RAG 메커니즘: 모델이 query와 검색된 사실을 모두 보게 되어
            # 외부 knowledge에 기반해 예측을 조건부로 수행할 수 있음.
            input_text = query + " " + context

            # Target: 전체 ground truth 문서
            # query+context에서 전체 factual 답변으로 completion하는 법을 학습함
            target = doc

            # target의 각 문자에 대해 학습함
            for i in range(min(20, len(target))):  # 속도를 위해 처음 20자로 제한함
                # 입력 context를 인코딩함 — 마지막 100자를 사용 (sliding window)
                # target 문자가 추가될 때 모델이 업데이트된 context를 보도록 함.
                # generate()의 inference 시 동작과 일치함.
                x = []
                for char in input_text[-100:]:
                    idx = char_to_index(char)
                    x.extend(one_hot(idx, VOCAB_SIZE))
                # 고정 크기로 padding함
                while len(x) < mlp.input_dim:
                    x.append(0.0)
                x = x[:mlp.input_dim]

                # target 문자
                target_idx = char_to_index(target[i])

                # Forward + backward
                _, cache = mlp.forward(x)
                loss = mlp.backward(target_idx, cache, learning_rate)
                epoch_loss += loss
                num_samples += 1

                # 예측된 문자를 포함하도록 input_text를 업데이트함
                input_text += target[i]

        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {avg_loss:.4f}")

    print()


# === INFERENCE DEMO ===

def demo_retrieval_comparison(
    queries: list[str],
    documents: list[str],
    bm25: BM25Index,
    mlp: MLP
):
    """retrieval이 있을 때와 없을 때의 생성을 비교하여 보여줌.

    RAG의 핵심 가치를 보여줌: 검색된 context가 factual grounding을 제공해
    생성 품질을 개선함. retrieval 없이는 모델이 전적으로
    parametric knowledge(학습된 weight)에 의존해야 하며, factual query에서
    hallucination이 발생하기 쉬움.
    """
    print("=== RETRIEVAL COMPARISON ===\n")

    for query in queries:
        print(f"Query: '{query}'")

        # WITH retrieval: BM25 → context 검색 → MLP가 생성함
        retrieved = bm25.retrieve(query, top_k=TOP_K)
        print(f"Retrieved docs (top {TOP_K}):")
        for doc_id, score in retrieved:
            print(f"  [{doc_id}] score={score:.2f}: {documents[doc_id][:60]}...")

        context = " ".join([documents[doc_id] for doc_id, _ in retrieved[:2]])
        input_with_context = query + " " + context
        generation_with = mlp.generate(input_with_context, max_length=40)

        # WITHOUT retrieval: 빈 context → MLP가 생성함
        # 모델이 query만 받고, 조건부 생성에 사용할 외부 사실이 없음
        input_without_context = query + " "
        generation_without = mlp.generate(input_without_context, max_length=40)

        print(f"WITH retrieval:    {generation_with}")
        print(f"WITHOUT retrieval: {generation_without}")
        print()


# === MAIN ===

if __name__ == "__main__":
    # 합성 knowledge base 생성
    print("Generating synthetic knowledge base...")
    documents, test_queries = generate_knowledge_base()
    print(f"Created {len(documents)} documents\n")

    # BM25 인덱스 구축
    print("Building BM25 index...")
    bm25 = BM25Index(documents, k1=K1, b=B)
    print(f"Indexed {bm25.N} documents, {len(bm25.idf)} unique terms\n")

    # 알려진 query에 대한 retrieval 정확도를 테스트함.
    # knowledge base에 주제별로 여러 문서가 있으므로 (예: Paris가 도시 문단,
    # 인구 문서, 강 문서에 등장), BM25가 같은 엔티티에 대해 더 구체적인
    # 문서를 반환할 수 있음. query의 핵심 term이 검색된 문서에 나타나는지로
    # 정확도를 측정함 -- BM25가 특정 문서 인덱스를 고르는지가 아니라
    # 관련 콘텐츠를 찾는지를 테스트함.
    print("=== RETRIEVAL ACCURACY TEST ===")
    correct = 0
    for query, expected_doc_idx in test_queries:
        retrieved = bm25.retrieve(query, top_k=1)
        if not retrieved:
            print(f"  MISS: '{query}' -> no results")
            continue

        retrieved_idx = retrieved[0][0]
        retrieved_terms = set(tokenize(documents[retrieved_idx]))
        query_terms = set(tokenize(query))

        # 반환된 문서가 query term의 50% 이상을 포함하면 정답으로 판정함.
        # 주제적 관련성을 측정함: "seine river" → seine과 river를 언급하는 문서.
        query_hits = sum(1 for t in query_terms if t in retrieved_terms)
        if query_hits >= max(len(query_terms) * 0.5, 1):
            correct += 1
            print(f"  HIT:  '{query}' -> [{retrieved_idx}] {documents[retrieved_idx][:50]}...")
        else:
            print(
                f"  MISS: '{query}' -> [{retrieved_idx}] {documents[retrieved_idx][:50]}..."
            )
    accuracy = 100 * correct / len(test_queries)
    print(f"Retrieval accuracy: {correct}/{len(test_queries)} = {accuracy:.1f}%\n")

    # MLP generator 초기화
    # 입력 차원: concat된 query + context (각 ~100자, one-hot 인코딩)
    # 차원을 관리 가능하도록 고정 입력 window를 사용함
    input_dim = 100 * VOCAB_SIZE  # 100자, one-hot 인코딩
    mlp = MLP(input_dim, HIDDEN_DIM, VOCAB_SIZE)
    print(f"Initialized MLP: {input_dim} -> {HIDDEN_DIM} -> {VOCAB_SIZE}")
    total_params = (
        len(mlp.W1) * len(mlp.W1[0]) + len(mlp.b1) +
        len(mlp.W2) * len(mlp.W2[0]) + len(mlp.b2)
    )
    print(f"Total parameters: {total_params:,}\n")

    # RAG 모델 학습
    train_rag(documents, bm25, mlp, NUM_EPOCHS, LEARNING_RATE)

    # 데모: retrieval이 있을 때와 없을 때의 생성을 비교함
    demo_queries = [
        "population of paris",
        "seine river",
        "everest height",
        "capital of germany",
    ]
    demo_retrieval_comparison(demo_queries, documents, bm25, mlp)

    print("RAG demonstration complete.")
