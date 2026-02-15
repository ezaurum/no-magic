"""
noise에서 이미지가 어떻게 생성되는지 -- Stable Diffusion의 기반인 denoising diffusion 알고리즘을
2D spiral로 시연함. noise를 예측하는 모델을 학습한 뒤, 순수한 랜덤에서 반복적으로 noise를 제거해
새로운 sample을 생성함.
"""
# Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (2020).
# https://arxiv.org/abs/2006.11239
# 이 2D 구현은 Stable Diffusion에서 사용되는 DDPM 알고리즘을 그대로 보존하되,
# 이미지에 대한 수십억 파라미터 U-Net을 point cloud에 대한 ~1000 파라미터 MLP로 축소함.

from __future__ import annotations

import math
import random

random.seed(42)


# === CONSTANTS ===

# Diffusion process hyperparameter
T = 100  # diffusion timestep 수 -- 프로덕션 모델은 1000을 사용하지만, 과도한 실행 시간 없이
# 알고리즘을 보여주기에는 100이면 충분함
BETA_START = 0.0001  # 초기 noise 수준 (매우 작음)
BETA_END = 0.02  # 최종 noise 수준 (적당한 corruption)
# 왜 linear schedule인가: 가장 단순한 옵션임; cosine schedule이 품질을 개선하지만
# 복잡성이 추가됨. 핵심 알고리즘을 가르치기에는 linear schedule로 충분함.

# 모델 아키텍처
HIDDEN_DIM = 64  # MLP hidden layer 크기 (~1000 총 파라미터)
TIME_EMB_DIM = 32  # sinusoidal timestep embedding 차원

# 학습
NUM_EPOCHS = 8000  # 학습 iteration (각각 하나의 랜덤 (x0, t) 쌍을 처리함)
# 6400개 파라미터에 8000번 업데이트 ≈ 파라미터당 1.25번 업데이트 — 부드러운
# 2D spiral에 충분함. 프로덕션 DDPM은 수백만 번 업데이트함.
LEARNING_RATE = 0.001  # Adam learning rate
NUM_SAMPLES = 800  # 학습 데이터 포인트 수 (2D spiral)

# 추론
NUM_GENERATED = 500  # 통계를 위해 생성할 sample 수


# === SYNTHETIC DATA GENERATION ===

def generate_spiral(num_points: int) -> list[tuple[float, float]]:
    """학습용 2D spiral point cloud를 생성함.

    spiral은 각도가 증가하면서 반지름이 선형적으로 커져, 통계로 시각적 검증이
    쉬운 비자명한 분포를 만듦 (평균이 원점 근처, 제한된 분산).
    모델이 Gaussian blob을 암기하는 게 아니라 구조를 학습하는지 테스트함.
    """
    points = []
    for i in range(num_points):
        # 파라메트릭 spiral: r = theta / (2*pi), 단위 반지름당 한 바퀴
        theta = (i / num_points) * 4 * math.pi  # 2바퀴 회전
        r = theta / (2 * math.pi)

        # 직교 좌표로 변환
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        # 현실적으로 만들기 위해 작은 Gaussian noise를 추가함 (완벽한 곡선이 아님)
        x += random.gauss(0, 0.05)
        y += random.gauss(0, 0.05)

        points.append((x, y))

    return points


# === NOISE SCHEDULE ===

def compute_noise_schedule(t_steps: int, beta_start: float, beta_end: float):
    """모든 timestep에 대한 noise schedule 계수를 미리 계산함.

    forward diffusion process는 각 timestep에서 다음에 따라 Gaussian noise를 추가함:
        q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)

    alpha_bar_t = prod(1 - beta_i for i in 1..t)를 미리 계산해
    순차 적용 없이 임의의 timestep으로 직접 noising할 수 있음:
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    이 closed-form 점프가 diffusion model 학습을 실용적으로 만드는 핵심 --
    t-1개의 이전 timestep을 순차적으로 거치지 않고 O(1)에 임의의 timestep을 샘플링할 수 있음.

    Returns:
        betas: 각 timestep의 noise variance (길이 T)
        alphas: 각 timestep의 1 - beta (길이 T)
        alpha_bars: alpha의 누적 곱 (길이 T)
        sqrt_alpha_bars: noising용 미리 계산된 sqrt (길이 T)
        sqrt_one_minus_alpha_bars: noise 계수용 미리 계산된 sqrt (길이 T)
    """
    # beta_start에서 beta_end까지 선형 보간
    betas = [beta_start + (beta_end - beta_start) * t / (t_steps - 1)
             for t in range(t_steps)]

    alphas = [1.0 - b for b in betas]

    # 누적 곱: alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t
    alpha_bars = []
    product = 1.0
    for alpha in alphas:
        product *= alpha
        alpha_bars.append(product)

    # forward process 수식을 위한 제곱근을 미리 계산함
    sqrt_alpha_bars = [math.sqrt(ab) for ab in alpha_bars]
    sqrt_one_minus_alpha_bars = [math.sqrt(1.0 - ab) for ab in alpha_bars]

    return betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars


# === TIMESTEP EMBEDDING ===

def sinusoidal_embedding(t: int, dim: int) -> list[float]:
    """sinusoidal positional encoding을 사용해 timestep t를 벡터로 인코딩함.

    차원 i에 대해:
        emb[2*i]   = sin(t / 10000^(2*i/dim))
        emb[2*i+1] = cos(t / 10000^(2*i/dim))

    왜 sinusoidal인가: 인접한 step 간 부드러운 보간과 함께 각 timestep에 대한
    고유한 표현을 제공함. 저주파 성분(초기 차원)은 t에 따라 천천히 변하고,
    고주파(후기 차원)는 빠르게 변함 -- 이 multi-scale encoding이
    모델이 인접한 timestep을 구별하는 데 도움이 됨.

    Transformer의 positional encoding과 동일한 embedding임 (Vaswani et al., 2017).
    """
    embedding = []
    for i in range(dim // 2):
        freq = 1.0 / (10000.0 ** (2 * i / dim))
        embedding.append(math.sin(t * freq))
        embedding.append(math.cos(t * freq))
    return embedding


# === NEURAL NETWORK (MANUAL IMPLEMENTATION) ===

def relu(x: float) -> float:
    """ReLU activation: max(0, x)."""
    return max(0.0, x)


def initialize_weights(input_dim: int, output_dim: int) -> list[list[float]]:
    """Xavier/Glorot uniform initialization으로 weight 행렬을 초기화함.

    Scale = sqrt(6 / (input_dim + output_dim))으로 레이어를 거쳐도 분산이
    대략 보존되어, 학습 초기에 gradient가 vanishing하거나 exploding하는 것을 방지함.
    """
    scale = math.sqrt(6.0 / (input_dim + output_dim))
    return [[random.uniform(-scale, scale) for _ in range(output_dim)]
            for _ in range(input_dim)]


def initialize_bias(dim: int) -> list[float]:
    """bias 벡터를 0으로 초기화함."""
    return [0.0 for _ in range(dim)]


class DenoisingMLP:
    """(noisy_data, timestep)이 주어지면 noise를 예측하는 소형 MLP.

    Architecture:
        Input: [x_noisy (2D), t_embedding (TIME_EMB_DIM)] -> concat하여 (2+TIME_EMB_DIM)D
        Hidden1: (2+TIME_EMB_DIM) -> HIDDEN_DIM, ReLU
        Hidden2: HIDDEN_DIM -> HIDDEN_DIM, ReLU
        Output: HIDDEN_DIM -> 2 (예측된 noise, activation 없음)

    프로덕션 diffusion model(Stable Diffusion)에서는 이 MLP가 수십억 파라미터의
    U-Net(attention layer, skip connection 포함)으로 대체됨.
    하지만 학습 목표는 동일함: x_t와 t가 주어지면 epsilon을 예측함.
    """

    def __init__(self):
        input_dim = 2 + TIME_EMB_DIM

        # Layer 1: input -> hidden
        self.w1 = initialize_weights(input_dim, HIDDEN_DIM)
        self.b1 = initialize_bias(HIDDEN_DIM)

        # Layer 2: hidden -> hidden
        self.w2 = initialize_weights(HIDDEN_DIM, HIDDEN_DIM)
        self.b2 = initialize_bias(HIDDEN_DIM)

        # Layer 3: hidden -> output (2D noise)
        self.w3 = initialize_weights(HIDDEN_DIM, 2)
        self.b3 = initialize_bias(2)

        # Adam optimizer 상태 (first moment, second moment)
        self.m = {'w1': [[0.0]*HIDDEN_DIM for _ in range(input_dim)],
                  'b1': [0.0]*HIDDEN_DIM,
                  'w2': [[0.0]*HIDDEN_DIM for _ in range(HIDDEN_DIM)],
                  'b2': [0.0]*HIDDEN_DIM,
                  'w3': [[0.0]*2 for _ in range(HIDDEN_DIM)],
                  'b3': [0.0]*2}

        self.v = {'w1': [[0.0]*HIDDEN_DIM for _ in range(input_dim)],
                  'b1': [0.0]*HIDDEN_DIM,
                  'w2': [[0.0]*HIDDEN_DIM for _ in range(HIDDEN_DIM)],
                  'b2': [0.0]*HIDDEN_DIM,
                  'w3': [[0.0]*2 for _ in range(HIDDEN_DIM)],
                  'b3': [0.0]*2}

        self.step = 0  # Adam timestep 카운터

    def forward(self, x_noisy: tuple[float, float], t: int) -> tuple[float, float]:
        """Forward pass: (noisy point, timestep) -> predicted noise.

        backprop을 위한 중간 activation을 반환함.
        """
        # noisy 데이터와 timestep embedding을 concat함
        t_emb = sinusoidal_embedding(t, TIME_EMB_DIM)
        input_vec = [x_noisy[0], x_noisy[1]] + t_emb

        # Layer 1
        h1 = [sum(input_vec[i] * self.w1[i][j] for i in range(len(input_vec))) + self.b1[j]
              for j in range(HIDDEN_DIM)]
        h1_relu = [relu(h) for h in h1]

        # Layer 2
        h2 = [sum(h1_relu[i] * self.w2[i][j] for i in range(HIDDEN_DIM)) + self.b2[j]
              for j in range(HIDDEN_DIM)]
        h2_relu = [relu(h) for h in h2]

        # Layer 3 (output, activation 없음)
        output = [sum(h2_relu[i] * self.w3[i][j] for i in range(HIDDEN_DIM)) + self.b3[j]
                  for j in range(2)]

        # backprop용 cache
        self.cache = {
            'input': input_vec,
            'h1': h1,
            'h1_relu': h1_relu,
            'h2': h2,
            'h2_relu': h2_relu,
            'output': output
        }

        return tuple(output)

    def backward_and_update(self, grad_output: tuple[float, float], lr: float):
        """MSE gradient를 backprop하고 Adam으로 weight를 업데이트함.

        모든 레이어를 통한 수동 gradient 계산. 프로덕션에서는 autograd 프레임워크
        (PyTorch, JAX)가 처리하지만, 수동으로 구현하면 메커니즘이 드러남.
        """
        # output layer에서의 gradient
        grad_out = list(grad_output)

        # layer 3을 통한 backprop (linear, activation 없음)
        grad_w3 = [[self.cache['h2_relu'][i] * grad_out[j] for j in range(2)]
                   for i in range(HIDDEN_DIM)]
        grad_b3 = grad_out
        grad_h2_relu = [sum(self.w3[i][j] * grad_out[j] for j in range(2))
                        for i in range(HIDDEN_DIM)]

        # ReLU를 통한 backprop (입력 <= 0이면 미분값 0, 아니면 1)
        grad_h2 = [grad_h2_relu[i] if self.cache['h2'][i] > 0 else 0.0
                   for i in range(HIDDEN_DIM)]

        # layer 2를 통한 backprop
        grad_w2 = [[self.cache['h1_relu'][i] * grad_h2[j] for j in range(HIDDEN_DIM)]
                   for i in range(HIDDEN_DIM)]
        grad_b2 = grad_h2
        grad_h1_relu = [sum(self.w2[i][j] * grad_h2[j] for j in range(HIDDEN_DIM))
                        for i in range(HIDDEN_DIM)]

        # ReLU를 통한 backprop
        grad_h1 = [grad_h1_relu[i] if self.cache['h1'][i] > 0 else 0.0
                   for i in range(HIDDEN_DIM)]

        # layer 1을 통한 backprop
        input_dim = len(self.cache['input'])
        grad_w1 = [[self.cache['input'][i] * grad_h1[j] for j in range(HIDDEN_DIM)]
                   for i in range(input_dim)]
        grad_b1 = grad_h1

        # Adam 업데이트
        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # 각 파라미터를 Adam으로 업데이트함
        def adam_update(param, grad, m, v):
            """단일 파라미터 배열에 Adam 업데이트 규칙을 적용함."""
            # First moment (gradient의 지수 이동 평균)
            for i in range(len(param)):
                if isinstance(param[i], list):
                    for j in range(len(param[i])):
                        m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad[i][j]
                        v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad[i][j] ** 2
                        m_hat = m[i][j] / (1 - beta1 ** self.step)
                        v_hat = v[i][j] / (1 - beta2 ** self.step)
                        param[i][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
                else:
                    m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
                    m_hat = m[i] / (1 - beta1 ** self.step)
                    v_hat = v[i] / (1 - beta2 ** self.step)
                    param[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        adam_update(self.w1, grad_w1, self.m['w1'], self.v['w1'])
        adam_update(self.b1, grad_b1, self.m['b1'], self.v['b1'])
        adam_update(self.w2, grad_w2, self.m['w2'], self.v['w2'])
        adam_update(self.b2, grad_b2, self.m['b2'], self.v['b2'])
        adam_update(self.w3, grad_w3, self.m['w3'], self.v['w3'])
        adam_update(self.b3, grad_b3, self.m['b3'], self.v['b3'])


# === FORWARD DIFFUSION PROCESS ===

def add_noise(x0: tuple[float, float], t: int,
              sqrt_alpha_bars: list[float],
              sqrt_one_minus_alpha_bars: list[float]) -> tuple[tuple[float, float],
                                                                tuple[float, float]]:
    """clean 데이터 포인트 x0에 timestep t에서 noise를 추가함.

    Math-to-code mapping:
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    epsilon ~ N(0, I)는 샘플링하는 noise임.

    왜 이 수식이 동작하는가: forward process는 어떤 데이터 분포든 표준 Gaussian으로
    점진적으로 변환하는 Markov chain임. T step 후,
    x_T ≈ N(0, I)이며 x_0에 무관함. sqrt 계수가 분산을 보존함:
    Var(x_t) = alpha_bar_t + (1 - alpha_bar_t) = 1.

    Returns:
        x_t: noise가 추가된 데이터 포인트
        epsilon: 추가된 noise (학습의 ground truth)
    """
    # 표준 Gaussian에서 noise를 샘플링함
    epsilon = (random.gauss(0, 1), random.gauss(0, 1))

    # closed-form noising 수식을 적용함
    coeff_signal = sqrt_alpha_bars[t]
    coeff_noise = sqrt_one_minus_alpha_bars[t]

    x_t = (coeff_signal * x0[0] + coeff_noise * epsilon[0],
           coeff_signal * x0[1] + coeff_noise * epsilon[1])

    return x_t, epsilon


# === TRAINING ===

def train(data: list[tuple[float, float]], model: DenoisingMLP,
          betas: list[float], alphas: list[float], alpha_bars: list[float],
          sqrt_alpha_bars: list[float], sqrt_one_minus_alpha_bars: list[float],
          num_epochs: int, lr: float):
    """noise를 예측하도록 denoising 모델을 학습함.

    학습 루프:
        1. 학습 세트에서 랜덤 데이터 포인트 x_0를 샘플링함
        2. [0, T-1]에서 랜덤 timestep t를 샘플링함
        3. noise epsilon ~ N(0, I)를 샘플링함
        4. x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon을 계산함
        5. epsilon_pred = model(x_t, t)로 예측함
        6. Loss = MSE(epsilon_pred, epsilon)
        7. Backprop하고 weight를 업데이트함

    왜 clean 데이터 대신 noise를 예측하는가: 경험적으로 noise epsilon을 예측하는 것이
    x_0를 예측하는 것보다 학습하기 쉬움. 직관적으로, noise가 데이터(복잡한 spiral 구조)보다
    더 단순함 (zero-mean Gaussian).

    왜 MSE loss인가: DDPM의 variational lower bound 유도 (Ho et al., 2020)에서
    학습된 reverse process와 true reverse process 간의 KL divergence를 최소화하면
    예측된 noise와 실제 noise 간의 MSE로 귀결됨. MSE가 maximum likelihood 학습을 위한
    올바른 loss임.
    """
    print(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # 랜덤 학습 sample
        x0 = random.choice(data)

        # 랜덤 timestep (0부터 T-1)
        t = random.randint(0, T - 1)

        # forward process로 noise를 추가함
        x_t, epsilon_true = add_noise(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)

        # noise를 예측함
        epsilon_pred = model.forward(x_t, t)

        # MSE loss
        loss = ((epsilon_pred[0] - epsilon_true[0]) ** 2 +
                (epsilon_pred[1] - epsilon_true[1]) ** 2) / 2

        # MSE의 gradient: d/d(pred) [(pred - true)^2 / 2] = (pred - true)
        grad_loss = (epsilon_pred[0] - epsilon_true[0],
                     epsilon_pred[1] - epsilon_true[1])

        # Backprop 및 업데이트
        model.backward_and_update(grad_loss, lr)

        # 진행 상황 출력
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>5}/{num_epochs}  Loss: {loss:.6f}")


# === SAMPLING (REVERSE PROCESS) ===

def sample(model: DenoisingMLP, betas: list[float], alphas: list[float],
           alpha_bars: list[float]) -> tuple[float, float]:
    """순수 noise를 반복적으로 denoising하여 새로운 2D 포인트를 생성함.

    Reverse process (sampling):
        x_T ~ N(0, I)에서 시작
        t = T-1, T-2, ..., 0에 대해:
            epsilon_pred = model(x_t, t)
            x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_pred)
                      + sigma_t * z
            여기서 t > 0이면 z ~ N(0, I), t = 0이면 z = 0

    Math-to-code mapping (DDPM 논문, Equation 11):
        mean = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_pred)
        variance = sigma_t^2 = beta_t (간소화; 아래 signpost 참조)
        x_{t-1} ~ N(mean, variance)

    Signpost: 전체 DDPM reverse variance는 더 복잡함 (beta_t와 다른 수식 사이를
    보간함). 여기서는 단순화를 위해 sigma_t = sqrt(beta_t)를 사용함.
    약간 더 높은 분산의 sample이 생성되지만 핵심 알고리즘은 보존됨.
    """
    # 순수 noise에서 시작함
    x = (random.gauss(0, 1), random.gauss(0, 1))

    # t = T-1부터 t = 0까지 반복적으로 denoise함
    for t in range(T - 1, -1, -1):
        # 현재 timestep에서 noise를 예측함
        epsilon_pred = model.forward(x, t)

        # p(x_{t-1} | x_t)의 평균을 계산함
        coeff = 1.0 / math.sqrt(alphas[t])
        noise_coeff = betas[t] / math.sqrt(1.0 - alpha_bars[t])

        mean_x = coeff * (x[0] - noise_coeff * epsilon_pred[0])
        mean_y = coeff * (x[1] - noise_coeff * epsilon_pred[1])

        # noise를 추가함 (t=0에서는 제외, 마지막 step은 결정적임)
        if t > 0:
            sigma = math.sqrt(betas[t])
            z = (random.gauss(0, 1), random.gauss(0, 1))
            x = (mean_x + sigma * z[0], mean_y + sigma * z[1])
        else:
            x = (mean_x, mean_y)

    return x


# === STATISTICS ===

def compute_statistics(points: list[tuple[float, float]]) -> dict[str, float]:
    """2D point cloud의 평균과 표준편차를 계산함."""
    n = len(points)

    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n

    var_x = sum((p[0] - mean_x) ** 2 for p in points) / n
    var_y = sum((p[1] - mean_y) ** 2 for p in points) / n

    std_x = math.sqrt(var_x)
    std_y = math.sqrt(var_y)

    return {
        'mean_x': mean_x,
        'mean_y': mean_y,
        'std_x': std_x,
        'std_y': std_y
    }


# === MAIN ===

if __name__ == "__main__":
    print("=" * 70)
    print("DENOISING DIFFUSION ON 2D SPIRAL")
    print("=" * 70)
    print()

    # 학습 데이터 생성
    print("Generating training data...")
    data = generate_spiral(NUM_SAMPLES)
    train_stats = compute_statistics(data)
    print(f"  Training set: {NUM_SAMPLES} points")
    print(f"  Mean: ({train_stats['mean_x']:.4f}, {train_stats['mean_y']:.4f})")
    print(f"  Std:  ({train_stats['std_x']:.4f}, {train_stats['std_y']:.4f})")
    print()

    # noise schedule을 미리 계산함
    print("Computing noise schedule...")
    betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = \
        compute_noise_schedule(T, BETA_START, BETA_END)
    print(f"  Timesteps: {T}")
    print(f"  Beta range: [{BETA_START:.6f}, {BETA_END:.6f}]")
    print(f"  Alpha_bar at T-1: {alpha_bars[-1]:.6f}")
    print()

    # 모델 초기화
    print("Initializing denoising model...")
    model = DenoisingMLP()
    print(f"  Architecture: (2+{TIME_EMB_DIM}) -> {HIDDEN_DIM} -> {HIDDEN_DIM} -> 2")
    print(f"  Parameters: ~{(2 + TIME_EMB_DIM) * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * 2}")
    print()

    # 학습
    train(data, model, betas, alphas, alpha_bars, sqrt_alpha_bars,
          sqrt_one_minus_alpha_bars, NUM_EPOCHS, LEARNING_RATE)
    print()

    # sample 생성
    print(f"Generating {NUM_GENERATED} samples from trained model...")
    generated = [sample(model, betas, alphas, alpha_bars)
                 for _ in range(NUM_GENERATED)]
    gen_stats = compute_statistics(generated)
    print(f"  Generated set: {NUM_GENERATED} points")
    print(f"  Mean: ({gen_stats['mean_x']:.4f}, {gen_stats['mean_y']:.4f})")
    print(f"  Std:  ({gen_stats['std_x']:.4f}, {gen_stats['std_y']:.4f})")
    print()

    # 분포 비교
    print("Distribution comparison:")
    print(f"  Training mean: ({train_stats['mean_x']:.4f}, {train_stats['mean_y']:.4f})")
    print(f"  Generated mean: ({gen_stats['mean_x']:.4f}, {gen_stats['mean_y']:.4f})")
    print()
    print(f"  Training std: ({train_stats['std_x']:.4f}, {train_stats['std_y']:.4f})")
    print(f"  Generated std: ({gen_stats['std_x']:.4f}, {gen_stats['std_y']:.4f})")
    print()

    # 품질 지표: 생성된 분포 vs 학습 분포 비교
    # 평균: 학습 std로 정규화한 절대 차이 (z-score)를 사용함.
    # 평균이 0 근처일 때 백분율 차이가 불안정하기 때문임.
    # 표준편차: std는 항상 양수이므로 백분율 차이가 적절함.
    mean_x_zscore = abs(gen_stats['mean_x'] - train_stats['mean_x']) / train_stats['std_x']
    mean_y_zscore = abs(gen_stats['mean_y'] - train_stats['mean_y']) / train_stats['std_y']
    std_x_diff = abs(gen_stats['std_x'] - train_stats['std_x']) / train_stats['std_x'] * 100
    std_y_diff = abs(gen_stats['std_y'] - train_stats['std_y']) / train_stats['std_y'] * 100

    print("Quality metrics:")
    print(f"  Mean shift (in std units):  X={mean_x_zscore:.2f}σ  Y={mean_y_zscore:.2f}σ")
    print(f"  Std deviation difference:   X={std_x_diff:.1f}%  Y={std_y_diff:.1f}%")
    print()

    # 성공 기준: mean shift < 0.5σ이고 std 차이 20% 이내
    success = mean_x_zscore < 0.5 and mean_y_zscore < 0.5 and std_x_diff < 20 and std_y_diff < 20
    if success:
        print("SUCCESS: Generated distribution matches training distribution.")
    else:
        print("PARTIAL: Generated distribution differs from training (may need more epochs).")
    print()

    print("=" * 70)
    print("ALGORITHM COMPLETE")
    print("=" * 70)
    print()
    print("What just happened:")
    print("  1. Generated a 2D spiral (non-trivial distribution)")
    print("  2. Trained a tiny MLP to predict noise at each diffusion timestep")
    print("  3. Sampled new points by starting from random noise and iteratively")
    print("     removing predicted noise for T steps")
    print()
    print("Mapping to image diffusion (Stable Diffusion, DALL-E):")
    print("  - 2D coordinates (x, y) -> RGB pixel values (R, G, B)")
    print("  - ~1000-param MLP -> ~1 billion-param U-Net with attention")
    print("  - 800 training points -> hundreds of millions of images")
    print("  - Gaussian noise on (x,y) -> Gaussian noise on (R,G,B)")
    print()
    print("The algorithm is identical. The scale is different.")
    print("This is how all modern image generation models work.")
