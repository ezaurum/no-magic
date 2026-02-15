"""
데이터의 압축된 생성적 표현을 학습하는 방법 — reparameterization trick의 이해,
외부 의존성 없이 순수 Python으로 구현함.
"""
# Reference: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013).
# https://arxiv.org/abs/1312.6114
# reparameterization trick (z = μ + σ * ε)은 VAE를 학습 가능하게 만든 핵심 기여임
# — 이전에는 sampling 연산이 gradient 흐름을 차단했음.

from __future__ import annotations

import math
import random

random.seed(42)


# === CONSTANTS ===

LATENT_DIM = 2          # latent space z의 크기. 쉬운 해석을 위해 2D임.
HIDDEN_DIM = 16         # encoder와 decoder MLP의 hidden layer 크기.
LEARNING_RATE = 0.001   # Adam learning rate.
BETA = 1.0              # ELBO에서 KL 가중치. β=1은 표준 VAE, β>1은 disentanglement를 촉진함.
NUM_EPOCHS = 1000       # 학습 iteration.
BATCH_SIZE = 16         # stochastic gradient descent를 위한 minibatch 크기.

# Signpost: 프로덕션 VAE는 이미지에 convolutional encoder/decoder를 사용함. 이 2D 데이터
# MLP는 복잡성 1%로 동일한 원리(ELBO, reparameterization, latent interpolation)를
# 보여줌. 알고리즘은 동일하며 -- 픽셀로 확장할 때 encoder/decoder 아키텍처만 변경됨.


# === SYNTHETIC DATA GENERATION ===

def generate_data(n_points: int = 800) -> list[list[float]]:
    """학습용 2D Gaussian mixture를 생성함.

    VAE가 학습할 흥미로운 구조를 갖도록 서로 다른 위치에 4개의 cluster를 만듦.
    단일 Gaussian이면 자명한 문제가 됨 (VAE가 평균/분산을 직접 학습할 수 있음).
    여러 mode가 latent space를 의미 있게 구성하도록 강제함.
    """
    # 2D 공간에서 대략 정사각형으로 배치된 4개의 cluster 중심.
    centers = [
        [-2.0, -2.0],
        [-2.0, 2.0],
        [2.0, -2.0],
        [2.0, 2.0],
    ]
    variance = 0.3  # cluster가 구분되지만 분리되지는 않을 정도의 작은 분산.

    data = []
    for _ in range(n_points):
        # cluster를 랜덤으로 선택한 뒤, N(center, variance)에서 샘플링함.
        center = random.choice(centers)
        x = random.gauss(center[0], math.sqrt(variance))
        y = random.gauss(center[1], math.sqrt(variance))
        data.append([x, y])

    return data


# === MLP UTILITIES ===
# plain float 배열을 사용함 (Value autograd 클래스가 아님). scalar autograd로 VAE를
# 학습하면 7분 실행 시간 제한에 걸리기 때문임. 수동 gradient 계산으로 핵심 VAE 알고리즘을
# 보이면서 실행 시간 제약을 충족함.

def matrix_multiply(a: list[list[float]], b: list[float]) -> list[float]:
    """행렬 a (m×n)와 벡터 b (n,)를 곱해 벡터 (m,)를 반환함."""
    return [sum(a[i][j] * b[j] for j in range(len(b))) for i in range(len(a))]


def relu(x: list[float]) -> list[float]:
    """ReLU activation: max(0, x) element-wise."""
    return [max(0.0, val) for val in x]


def relu_grad(x: list[float]) -> list[float]:
    """ReLU의 gradient: x > 0이면 1, 아니면 0."""
    return [1.0 if val > 0 else 0.0 for val in x]


def add_bias(x: list[float], b: list[float]) -> list[float]:
    """bias 벡터 b를 x에 element-wise로 더함."""
    return [x[i] + b[i] for i in range(len(x))]


def init_weights(rows: int, cols: int) -> list[list[float]]:
    """Xavier/Glorot initialization으로 weight를 초기화함.

    sqrt(2 / (rows + cols))로 스케일링해 안정적인 gradient를 유지함. 이것 없이는
    deep network에서 activation이 vanishing/exploding함.
    """
    scale = math.sqrt(2.0 / (rows + cols))
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def init_bias(size: int) -> list[float]:
    """bias를 0으로 초기화함 (표준 관례)."""
    return [0.0 for _ in range(size)]


# === ENCODER MLP ===

def encoder_forward(
    x: list[float],
    w1: list[list[float]],
    b1: list[float],
    w_mean: list[list[float]],
    b_mean: list[float],
    w_logvar: list[list[float]],
    b_logvar: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """Encoder: input (2D) → hidden (ReLU) → latent space의 (mean, log_var).

    왜 두 개의 출력 head(mean과 log_var)인가? encoder가 approximate posterior
    q(z|x)를 Gaussian으로 파라미터화하기 때문임. 분산 σ²를 직접 출력하지 않고
    log(σ²)를 출력하는 이유:
    - 분산은 양수여야 하지만 네트워크 출력은 제약이 없음
    - log_var를 최적화하면 수치적 문제를 피할 수 있음 (exp는 항상 양수)
    - variational inference의 표준 파라미터화임

    Returns: (hidden_state, mean, log_var), mean과 log_var 모두 (latent_dim,) 형태임
    """
    # Input → ReLU activation이 있는 hidden layer
    hidden = relu(add_bias(matrix_multiply(w1, x), b1))

    # Hidden → latent 분포의 mean (제약 없음)
    mean = add_bias(matrix_multiply(w_mean, hidden), b_mean)

    # Hidden → latent 분포의 log variance (제약 없음)
    # reparameterization trick 계산 시 이것을 exponentiate하므로,
    # log_var는 어떤 실수든 가능하며, exp(0.5 * log_var) = σ는 양수가 됨.
    log_var = add_bias(matrix_multiply(w_logvar, hidden), b_logvar)

    return hidden, mean, log_var


# === REPARAMETERIZATION TRICK ===
# 이 스크립트의 핵심 교육 포인트. 나머지는 모두 기반 구조이며,
# 이 하나의 함수가 VAE를 학습 가능하게 만드는 것임.

def reparameterize(mean: list[float], log_var: list[float]) -> list[float]:
    """reparameterization trick을 통해 q(z|x)에서 z를 샘플링함.

    핵심 통찰 — 왜 이것이 동작하는가:

    z ~ N(μ, σ²)를 샘플링하려 함. μ = encoder_mean(x), σ² = exp(encoder_log_var(x)).
    단순하게 하면:
        z = random.gauss(mean, sigma)

    하지만 이렇게 하면 gradient 흐름이 끊김. 랜덤성이 backpropagation을 차단함 —
    "랜덤 수를 샘플링"하는 연산의 미분이 정의되지 않기 때문에
    gradient가 sampling 연산을 통과할 수 없음.

    reparameterization trick이 이를 해결함:
        ε ~ N(0,1)           # 표준 정규분포에서 샘플링 (파라미터 없음)
        σ = exp(0.5 * log_var)    # log_var의 결정적 함수
        z = μ + σ * ε        # μ, log_var, 외부 ε의 결정적 함수

    이제 랜덤성(ε)이 computation graph 외부에 있음. gradient가
    μ와 log_var(결정적 네트워크 출력)를 통해 흐르지만, ε을 통해서는 흐르지 않음.
    이것이 sampling 연산을 미분 가능하게 만듦.

    Math-to-code mapping:
        μ: mean (encoder 출력)
        log(σ²): log_var (encoder 출력)
        σ: exp(0.5 * log_var)
        ε: epsilon (외부에서 샘플링됨)
        z: mean + sigma * epsilon

    Kingma & Welling (2013) 이전에는 REINFORCE 스타일 gradient estimator를 사용했는데
    분산이 훨씬 높아 더 많은 sample이 필요했음. reparameterization trick이
    VAE를 실용적으로 만든 것임.
    """
    epsilon = [random.gauss(0, 1) for _ in range(len(mean))]

    # σ = exp(0.5 * log_var). log_var = log(σ²)이므로
    # 0.5 * log_var = log(σ)이기 때문에 0.5 * log_var를 사용함.
    sigma = [math.exp(0.5 * lv) for lv in log_var]

    # z = μ + σ * ε
    z = [mean[i] + sigma[i] * epsilon[i] for i in range(len(mean))]

    return z


# === DECODER MLP ===

def decoder_forward(
    z: list[float],
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
) -> tuple[list[float], list[float]]:
    """Decoder: latent z → hidden (ReLU) → 복원된 output (2D).

    Returns: (hidden_state, output), output은 복원된 2D 포인트임.
    """
    # Latent → ReLU activation이 있는 hidden layer
    hidden = relu(add_bias(matrix_multiply(w1, z), b1))

    # Hidden → output (2D 복원 포인트, activation 없음)
    # 데이터가 제약 없으므로 (음수일 수 있음) activation을 적용하지 않음.
    output = add_bias(matrix_multiply(w2, hidden), b2)

    return hidden, output


# === ELBO LOSS ===

def compute_loss(
    x: list[float],
    mean: list[float],
    log_var: list[float],
    x_recon: list[float],
    beta: float,
) -> tuple[float, float, float]:
    """Evidence Lower Bound (ELBO) loss를 계산함.

    ELBO = reconstruction_loss + β * KL_divergence

    왜 이 loss 함수인가:
    VAE는 데이터의 log-likelihood log p(x)를 최대화함. 이를 직접 계산할 수 없으므로
    대신 하한(ELBO)을 최대화함. ELBO 최대화 ≈ log p(x) 최대화.

    ELBO는 두 항으로 분해됨:
    1. Reconstruction loss: decoder가 z에서 x를 얼마나 잘 복원하는지
       MSE (mean squared error)를 사용함: ||x - decoder(z)||²
    2. KL divergence: q(z|x)가 prior p(z) = N(0,I)와 얼마나 다른지
       latent space가 부드럽고 연속적이도록 정규화함.

    왜 KL divergence인가? latent space에 좋은 성질을 강제함:
    - 평균이 0 근처, 분산이 1 근처 (prior와 일치)
    - 인접한 z 값 사이의 부드러운 전이
    - 추론 시 N(0,I)에서 샘플링해 decode하면 새로운 데이터를 생성할 수 있음

    KL 정규화 없이는 encoder가 임의의 불연속적 매핑을 학습하고
    (예: cluster 1 → z=[100,0], cluster 2 → z=[-50,200]) decoder가
    overfit하게 됨. N(0,1)에서의 랜덤 sample이 쓸모없는 결과를 decode하므로
    latent space가 생성에 쓸모없게 됨.
    """
    # Reconstruction loss: input과 복원된 output 사이의 MSE
    reconstruction_loss = sum((x[i] - x_recon[i]) ** 2 for i in range(len(x)))

    # 대각 Gaussian에 대한 KL divergence KL(q(z|x) || p(z)).
    # q와 p가 모두 Gaussian이면, KL이 closed form을 가짐 (sampling 불필요):
    #   KL(N(μ, σ²) || N(0,I)) = 0.5 * sum(1 + log(σ²) - μ² - σ²)
    #                           = 0.5 * sum(1 + log_var - mean² - exp(log_var))
    #
    # Math-to-code mapping:
    #   μ: mean
    #   σ²: exp(log_var)
    #   log(σ²): log_var
    #
    # 왜 closed form인가: 두 분포가 모두 Gaussian이고, 두 Gaussian 사이의 KL은
    # 해석적임 (적분 계산 불필요).
    #
    # KL clamping: exp(log_var) 폭발을 방지하기 위해 log_var를 [-5, 5]로 clamp함.
    # exp(5) = 148 (합리적인 분산); exp(10) = 22,026 (KL이 폭발하고 gradient가
    # vanish함). clamping 없이는 encoder가 극단적인 log_var 값을 출력해
    # 수치적 불안정성을 유발할 수 있음.
    kl_loss = 0.0
    for i in range(len(mean)):
        # 수치적 폭발을 방지하기 위해 log_var를 clamp함
        clamped_lv = max(min(log_var[i], 5.0), -5.0)
        kl_loss += 1.0 + clamped_lv - mean[i] ** 2 - math.exp(clamped_lv)
    kl_loss = -0.5 * kl_loss  # 수식을 마이너스 부호로 유도했으므로 음수를 취함

    # 총 ELBO loss (negative ELBO를 최소화하며, 이는 ELBO를 최대화하는 것과 동일함)
    # β-weighting: β=1은 표준 VAE. β>1 (β-VAE)은 KL에 더 큰 페널티를 줘
    # 복원 품질과 latent space 구조 사이의 trade-off로 disentangled representation을
    # 촉진함. β<1은 복원을 강조하는 대신 latent space가 덜 정돈됨.
    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


# === MANUAL GRADIENT COMPUTATION ===

def backward_and_update(
    x: list[float],
    mean: list[float],
    log_var: list[float],
    z: list[float],
    x_recon: list[float],
    enc_hidden: list[float],
    dec_hidden: list[float],
    # Encoder weights
    enc_w1: list[list[float]],
    enc_b1: list[float],
    enc_w_mean: list[list[float]],
    enc_b_mean: list[float],
    enc_w_logvar: list[list[float]],
    enc_b_logvar: list[float],
    # Decoder weights
    dec_w1: list[list[float]],
    dec_b1: list[float],
    dec_w2: list[list[float]],
    dec_b2: list[float],
    # Adam moments
    m_enc_w1: list[list[float]],
    v_enc_w1: list[list[float]],
    m_enc_b1: list[float],
    v_enc_b1: list[float],
    m_enc_w_mean: list[list[float]],
    v_enc_w_mean: list[list[float]],
    m_enc_b_mean: list[float],
    v_enc_b_mean: list[float],
    m_enc_w_logvar: list[list[float]],
    v_enc_w_logvar: list[list[float]],
    m_enc_b_logvar: list[float],
    v_enc_b_logvar: list[float],
    m_dec_w1: list[list[float]],
    v_dec_w1: list[list[float]],
    m_dec_b1: list[float],
    v_dec_b1: list[float],
    m_dec_w2: list[list[float]],
    v_dec_w2: list[list[float]],
    m_dec_b2: list[float],
    v_dec_b2: list[float],
    lr: float,
    beta: float,
) -> None:
    """gradient를 계산하고 Adam optimizer로 파라미터를 업데이트함.

    이 함수는 의도적으로 김 — reconstruction loss와 KL divergence에서
    decoder, reparameterization, encoder를 거쳐 전체 gradient 흐름을 보여줌.
    reparameterization trick gradient가 핵심 통찰임.
    """
    # --- 복원된 output에 대한 reconstruction loss의 gradient ---
    # d(MSE)/d(x_recon) = 2 * (x_recon - x)
    grad_recon = [2.0 * (x_recon[i] - x[i]) for i in range(len(x))]

    # --- decoder를 통한 backprop ---
    # Decoder output layer: x_recon = dec_w2 @ dec_hidden + dec_b2
    grad_dec_b2 = grad_recon[:]
    grad_dec_w2 = [[grad_recon[i] * dec_hidden[j] for j in range(len(dec_hidden))]
                   for i in range(len(grad_recon))]
    grad_dec_hidden = [sum(dec_w2[i][j] * grad_recon[i] for i in range(len(grad_recon)))
                       for j in range(len(dec_hidden))]

    # Decoder hidden layer: dec_hidden = ReLU(dec_w1 @ z + dec_b1)
    grad_dec_hidden = [grad_dec_hidden[i] * relu_grad([dec_hidden[i]])[0]
                       for i in range(len(grad_dec_hidden))]
    grad_dec_b1 = grad_dec_hidden[:]
    grad_dec_w1 = [[grad_dec_hidden[i] * z[j] for j in range(len(z))]
                   for i in range(len(grad_dec_hidden))]
    grad_z_recon = [sum(dec_w1[i][j] * grad_dec_hidden[i] for i in range(len(grad_dec_hidden)))
                    for j in range(len(z))]

    # --- mean과 log_var에 대한 KL divergence의 gradient ---
    # KL = -0.5 * sum(1 + log_var - mean² - exp(log_var))
    # d(KL)/d(mean) = -0.5 * (-2 * mean) = mean
    # d(KL)/d(log_var) = -0.5 * (1 - exp(log_var))
    grad_mean_kl = [beta * mean[i] for i in range(len(mean))]
    grad_logvar_kl = [beta * -0.5 * (1.0 - math.exp(max(min(log_var[i], 5.0), -5.0)))
                      for i in range(len(log_var))]

    # --- reparameterization trick을 통한 gradient ---
    # z = mean + exp(0.5 * log_var) * epsilon
    # d(loss)/d(mean) = d(loss)/d(z) * d(z)/d(mean) + d(KL)/d(mean)
    #                 = d(loss)/d(z) * 1 + d(KL)/d(mean)
    # d(loss)/d(log_var) = d(loss)/d(z) * d(z)/d(log_var) + d(KL)/d(log_var)
    #                    = d(loss)/d(z) * (0.5 * exp(0.5*log_var) * epsilon) + d(KL)/d(log_var)
    epsilon = [(z[i] - mean[i]) / (math.exp(0.5 * log_var[i]) + 1e-10) for i in range(len(z))]

    grad_mean = [grad_z_recon[i] + grad_mean_kl[i] for i in range(len(mean))]
    grad_logvar = [grad_z_recon[i] * 0.5 * math.exp(0.5 * log_var[i]) * epsilon[i] + grad_logvar_kl[i]
                   for i in range(len(log_var))]

    # --- encoder를 통한 backprop ---
    # Encoder mean head: mean = enc_w_mean @ enc_hidden + enc_b_mean
    grad_enc_b_mean = grad_mean[:]
    grad_enc_w_mean = [[grad_mean[i] * enc_hidden[j] for j in range(len(enc_hidden))]
                       for i in range(len(grad_mean))]
    grad_enc_hidden_mean = [sum(enc_w_mean[i][j] * grad_mean[i] for i in range(len(grad_mean)))
                            for j in range(len(enc_hidden))]

    # Encoder log_var head: log_var = enc_w_logvar @ enc_hidden + enc_b_logvar
    grad_enc_b_logvar = grad_logvar[:]
    grad_enc_w_logvar = [[grad_logvar[i] * enc_hidden[j] for j in range(len(enc_hidden))]
                         for i in range(len(grad_logvar))]
    grad_enc_hidden_logvar = [sum(enc_w_logvar[i][j] * grad_logvar[i] for i in range(len(grad_logvar)))
                              for j in range(len(enc_hidden))]

    # 두 head의 gradient를 합침
    grad_enc_hidden = [grad_enc_hidden_mean[i] + grad_enc_hidden_logvar[i]
                       for i in range(len(enc_hidden))]

    # Encoder hidden layer: enc_hidden = ReLU(enc_w1 @ x + enc_b1)
    grad_enc_hidden = [grad_enc_hidden[i] * relu_grad([enc_hidden[i]])[0]
                       for i in range(len(grad_enc_hidden))]
    grad_enc_b1 = grad_enc_hidden[:]
    grad_enc_w1 = [[grad_enc_hidden[i] * x[j] for j in range(len(x))]
                   for i in range(len(grad_enc_hidden))]

    # --- Adam 업데이트 ---
    # Adam: first moment와 second moment 추정치를 사용해 파라미터별 적응적 learning rate를 적용함.
    # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    # θ_t = θ_{t-1} - α * m_t / (sqrt(v_t) + ε)
    #
    # ε는 v (second moment)가 0에 가까울 때 0으로 나누는 것을 방지함.
    # 표준 hyperparameter: β₁=0.9, β₂=0.999, ε=1e-8 (PyTorch/TensorFlow과 동일).
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # 단일 파라미터를 Adam으로 업데이트하는 helper
    def adam_update(param, grad, m, v):
        for i in range(len(param)):
            if isinstance(param[i], list):  # weight 행렬
                for j in range(len(param[i])):
                    m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad[i][j]
                    v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad[i][j] ** 2
                    param[i][j] -= lr * m[i][j] / (math.sqrt(v[i][j]) + eps)
            else:  # bias 벡터
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
                param[i] -= lr * m[i] / (math.sqrt(v[i]) + eps)

    # encoder 파라미터 업데이트
    adam_update(enc_w1, grad_enc_w1, m_enc_w1, v_enc_w1)
    adam_update(enc_b1, grad_enc_b1, m_enc_b1, v_enc_b1)
    adam_update(enc_w_mean, grad_enc_w_mean, m_enc_w_mean, v_enc_w_mean)
    adam_update(enc_b_mean, grad_enc_b_mean, m_enc_b_mean, v_enc_b_mean)
    adam_update(enc_w_logvar, grad_enc_w_logvar, m_enc_w_logvar, v_enc_w_logvar)
    adam_update(enc_b_logvar, grad_enc_b_logvar, m_enc_b_logvar, v_enc_b_logvar)

    # decoder 파라미터 업데이트
    adam_update(dec_w1, grad_dec_w1, m_dec_w1, v_dec_w1)
    adam_update(dec_b1, grad_dec_b1, m_dec_b1, v_dec_b1)
    adam_update(dec_w2, grad_dec_w2, m_dec_w2, v_dec_w2)
    adam_update(dec_b2, grad_dec_b2, m_dec_b2, v_dec_b2)


# === TRAINING LOOP ===

if __name__ == "__main__":
    print("Generating synthetic 2D data (mixture of 4 Gaussians)...")
    data = generate_data()
    print(f"Generated {len(data)} 2D points\n")

    # encoder weight 초기화
    enc_w1 = init_weights(HIDDEN_DIM, 2)          # 2D input → hidden
    enc_b1 = init_bias(HIDDEN_DIM)
    enc_w_mean = init_weights(LATENT_DIM, HIDDEN_DIM)  # hidden → mean
    enc_b_mean = init_bias(LATENT_DIM)
    enc_w_logvar = init_weights(LATENT_DIM, HIDDEN_DIM)  # hidden → log_var
    enc_b_logvar = init_bias(LATENT_DIM)

    # decoder weight 초기화
    dec_w1 = init_weights(HIDDEN_DIM, LATENT_DIM)  # latent → hidden
    dec_b1 = init_bias(HIDDEN_DIM)
    dec_w2 = init_weights(2, HIDDEN_DIM)          # hidden → 2D output
    dec_b2 = init_bias(2)

    # Adam moment 버퍼 초기화 (모두 0)
    def init_moments_like(shape):
        if isinstance(shape[0], list):  # 행렬
            return [[0.0 for _ in range(len(shape[0]))] for _ in range(len(shape))]
        else:  # 벡터
            return [0.0 for _ in range(len(shape))]

    m_enc_w1, v_enc_w1 = init_moments_like(enc_w1), init_moments_like(enc_w1)
    m_enc_b1, v_enc_b1 = init_moments_like(enc_b1), init_moments_like(enc_b1)
    m_enc_w_mean, v_enc_w_mean = init_moments_like(enc_w_mean), init_moments_like(enc_w_mean)
    m_enc_b_mean, v_enc_b_mean = init_moments_like(enc_b_mean), init_moments_like(enc_b_mean)
    m_enc_w_logvar, v_enc_w_logvar = init_moments_like(enc_w_logvar), init_moments_like(enc_w_logvar)
    m_enc_b_logvar, v_enc_b_logvar = init_moments_like(enc_b_logvar), init_moments_like(enc_b_logvar)

    m_dec_w1, v_dec_w1 = init_moments_like(dec_w1), init_moments_like(dec_w1)
    m_dec_b1, v_dec_b1 = init_moments_like(dec_b1), init_moments_like(dec_b1)
    m_dec_w2, v_dec_w2 = init_moments_like(dec_w2), init_moments_like(dec_w2)
    m_dec_b2, v_dec_b2 = init_moments_like(dec_b2), init_moments_like(dec_b2)

    print("Training VAE...")
    print(f"{'Epoch':<8} {'Total Loss':<12} {'Recon Loss':<12} {'KL Loss':<12}")
    print("-" * 48)

    for epoch in range(NUM_EPOCHS):
        # stochastic gradient descent를 위해 데이터를 셔플함
        random.shuffle(data)

        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0

        # 데이터를 minibatch로 처리함
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]

            batch_total_loss = 0.0
            batch_recon_loss = 0.0
            batch_kl_loss = 0.0

            for x in batch:
                # Forward pass
                enc_hidden, mean, log_var = encoder_forward(
                    x, enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar
                )
                z = reparameterize(mean, log_var)
                dec_hidden, x_recon = decoder_forward(z, dec_w1, dec_b1, dec_w2, dec_b2)

                # Loss 계산
                total_loss, recon_loss, kl_loss = compute_loss(x, mean, log_var, x_recon, BETA)

                batch_total_loss += total_loss
                batch_recon_loss += recon_loss
                batch_kl_loss += kl_loss

                # Backward pass 및 업데이트
                backward_and_update(
                    x, mean, log_var, z, x_recon, enc_hidden, dec_hidden,
                    enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar,
                    dec_w1, dec_b1, dec_w2, dec_b2,
                    m_enc_w1, v_enc_w1, m_enc_b1, v_enc_b1,
                    m_enc_w_mean, v_enc_w_mean, m_enc_b_mean, v_enc_b_mean,
                    m_enc_w_logvar, v_enc_w_logvar, m_enc_b_logvar, v_enc_b_logvar,
                    m_dec_w1, v_dec_w1, m_dec_b1, v_dec_b1,
                    m_dec_w2, v_dec_w2, m_dec_b2, v_dec_b2,
                    LEARNING_RATE, BETA,
                )

            # batch 평균 loss
            batch_total_loss /= len(batch)
            batch_recon_loss /= len(batch)
            batch_kl_loss /= len(batch)

            epoch_total_loss += batch_total_loss
            epoch_recon_loss += batch_recon_loss
            epoch_kl_loss += batch_kl_loss

        # 전체 batch 평균 loss
        num_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
        epoch_total_loss /= num_batches
        epoch_recon_loss /= num_batches
        epoch_kl_loss /= num_batches

        # 100 epoch마다 진행 상황 출력
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"{epoch + 1:<8} {epoch_total_loss:<12.4f} {epoch_recon_loss:<12.4f} {epoch_kl_loss:<12.4f}")

    print("\nTraining complete\n")

    # === INFERENCE DEMO ===

    print("=" * 60)
    print("INFERENCE: Latent Space Interpolation")
    print("=" * 60)
    print("Encode two data points, interpolate in latent space, decode.\n")

    # 서로 다른 cluster에서 두 포인트를 선택함
    point_a = data[0]      # 한 cluster에서 온 것으로 추정됨
    point_b = data[200]    # 다른 cluster에서 온 것으로 추정됨

    # 두 포인트를 encode함
    _, mean_a, log_var_a = encoder_forward(
        point_a, enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar
    )
    _, mean_b, log_var_b = encoder_forward(
        point_b, enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar
    )

    print(f"Point A: {[round(v, 3) for v in point_a]}")
    print(f"  → Latent mean: {[round(v, 3) for v in mean_a]}")
    print(f"Point B: {[round(v, 3) for v in point_b]}")
    print(f"  → Latent mean: {[round(v, 3) for v in mean_b]}\n")

    print("Interpolation (5 steps from A to B):")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # latent space에서 선형 보간함
        z_interp = [mean_a[i] * (1 - alpha) + mean_b[i] * alpha for i in range(LATENT_DIM)]

        # 보간된 latent 포인트를 decode함
        _, x_interp = decoder_forward(z_interp, dec_w1, dec_b1, dec_w2, dec_b2)

        print(f"  α={alpha:.2f}: z={[round(v, 3) for v in z_interp]} → x={[round(v, 3) for v in x_interp]}")

    print()

    print("=" * 60)
    print("INFERENCE: Prior Sampling (Generation)")
    print("=" * 60)
    print("Sample z ~ N(0,1), decode to generate new data points.\n")

    generated_points = []
    for _ in range(10):
        # prior N(0,1)에서 샘플링함
        z_sample = [random.gauss(0, 1) for _ in range(LATENT_DIM)]

        # decode해 새로운 2D 포인트를 생성함
        _, x_gen = decoder_forward(z_sample, dec_w1, dec_b1, dec_w2, dec_b2)

        generated_points.append(x_gen)

    print("10 generated points:")
    for i, point in enumerate(generated_points):
        print(f"  {i + 1}. {[round(v, 3) for v in point]}")

    print()

    print("=" * 60)
    print("INFERENCE: Reconstruction Quality")
    print("=" * 60)
    print("Encode training points, decode them, compare original vs reconstructed.\n")

    print("Original → Reconstructed (5 samples):")
    for i in range(5):
        x_orig = data[i * 100]  # 100번째 포인트마다 샘플링함

        # encode하고 decode함
        _, mean, log_var = encoder_forward(
            x_orig, enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar
        )
        z = mean  # 복원 품질 확인을 위해 mean을 사용 (sampling 없음)
        _, x_rec = decoder_forward(z, dec_w1, dec_b1, dec_w2, dec_b2)

        # 복원 오차를 계산함
        error = math.sqrt(sum((x_orig[j] - x_rec[j]) ** 2 for j in range(len(x_orig))))

        print(f"  {[round(v, 3) for v in x_orig]} → {[round(v, 3) for v in x_rec]} (error: {error:.4f})")

    print()
    print("VAE training and inference complete.")
