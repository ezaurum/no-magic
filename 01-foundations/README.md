# Foundations

Core algorithms that form the building blocks of modern AI systems. These are the primitives — if you understand these, everything else is composition.

## Scripts

| Script              | Algorithm                                                | Status    |
| ------------------- | -------------------------------------------------------- | --------- |
| `03-microgpt.py`       | Autoregressive language model (GPT) with scalar autograd | Complete  |
| `04-micrornn.py`       | Vanilla RNN vs. GRU — vanishing gradients and gating     | Complete  |
| `01-microtokenizer.py` | Byte-Pair Encoding (BPE) tokenization                    | Complete  |
| `02-microembedding.py` | Contrastive embedding learning (InfoNCE)                 | Complete  |
| `05-microrag.py`       | Retrieval-Augmented Generation (BM25 + MLP)              | Complete  |
| `06-microdiffusion.py` | Denoising diffusion on 2D point clouds                   | Complete  |
| `07-microvae.py`       | Variational Autoencoder with reparameterization trick    | Complete  |

## Future Candidates

These algorithms are strong candidates for future addition. Each would need to meet the project constraints (single file, zero dependencies, trains and infers, under 7 minutes on CPU).

| Algorithm                            | What It Would Teach                                 | Notes                                                                     |
| ------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------- |
| **LSTM**                             | Long Short-Term Memory gating (3 gates vs. GRU's 2) | Could extend 04-micrornn.py or be standalone                                 |
| **GAN**                              | Generative Adversarial Networks on 2D data          | Generator vs. discriminator dynamics, mode collapse, training instability |
| **Transformer Encoder (BERT-style)** | Masked language modeling, bidirectional attention   | Contrasts with microgpt (decoder-only)                                    |
| **ConvNet**                          | Convolution from scratch on tiny images             | Kernel sliding, pooling, feature maps — the vision primitive              |
| **Optimizer Comparison**             | SGD vs. Momentum vs. Adam side-by-side              | Convergence dynamics, adaptive learning rates                             |
| **Word2Vec**                         | Skip-gram with negative sampling                    | Classic embedding algorithm, simpler than contrastive learning            |

## Learning Path

For a guided walkthrough of the foundations tier, follow this order:

```plaintext
01-microtokenizer.py   → How text becomes numbers
02-microembedding.py   → How meaning becomes geometry
03-microgpt.py         → How sequences become predictions
05-microrag.py         → How retrieval augments generation
04-micrornn.py         → How sequences were modeled before attention
06-microdiffusion.py   → How data emerges from noise
07-microvae.py         → How to learn compressed generative representations
```
