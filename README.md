# Model Distillation

Knowledge distillation from [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m) (teacher) into [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m) (student) on raw web text from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), set up to run on a single Tesla P100.

## How it works

The student is trained on a mix of two losses:

- **Language-modeling loss** – the student's normal next-token prediction loss on the text.
- **Distillation loss** – KL divergence between the teacher's and student's next-token distributions, softened by a temperature.

$$
\mathcal{L} = \alpha \, \mathcal{L}_{\text{LM}} + (1 - \alpha) \, T^2 \, \mathcal{L}_{\text{KD}}
$$

With the defaults ($\alpha = 0.5$, $T = 2$), both terms are weighted equally.

### Notation

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $V$ | Vocabulary size (50,304 for Pythia) |
| $x_t$ | Token at position $t$ |
| $z_{t} \in \mathbb{R}^V$ | Student logits at position $t$ |
| $v_{t} \in \mathbb{R}^V$ | Teacher logits at position $t$ |
| $T$ | Distillation temperature (`temperature`) |
| $\alpha$ | Weight on the language-modeling loss (`alpha`) |

### Softmax with temperature

Logits are turned into probabilities over the vocabulary at every position. Dividing by $T$ before the softmax flattens the distribution, so lower-ranked tokens get more probability:

$$
p_{t,i} = \frac{\exp(z_{t,i} / T)}{\sum_{j=1}^{V} \exp(z_{t,j} / T)}
\qquad
q_{t,i} = \frac{\exp(v_{t,i} / T)}{\sum_{j=1}^{V} \exp(v_{t,j} / T)}
$$

$p$ is the student's softened distribution and $q$ is the teacher's. $T = 1$ is the ordinary softmax; $T > 1$ is softer.

### Language-modeling loss

Standard next-token cross-entropy on the student, computed by the model from its unsoftened logits ($T = 1$). Labels are a copy of the input shifted by one; positions whose label is `-100` (padding) are excluded:

$$
\mathcal{L}_{\text{LM}} = -\frac{1}{|\mathcal{M}|} \sum_{(b,t) \in \mathcal{M}} \log P_{\text{student}}\left(x_{b,t+1} \mid x_{b,1}, \dots, x_{b,t}\right)
$$

where $\mathcal{M}$ is the set of non-padding positions in the batch, and $P_{\text{student}}$ is the ordinary softmax of the student's logits $z_{b,t}$ (no temperature).

### Distillation loss

KL divergence from the teacher's softened distribution to the student's, summed over the vocabulary and positions and divided by the batch size (PyTorch's `KLDivLoss(reduction="batchmean")`):

$$
\mathcal{L}_{\text{KD}} = \frac{1}{B} \sum_{b=1}^{B} \sum_{t=1}^{L} \sum_{i=1}^{V} q_{b,t,i} \, \log \frac{q_{b,t,i}}{p_{b,t,i}}
$$

Two consequences of this exact form:

- It is a **per-sequence sum**, not a per-token average, so its scale grows with sequence length.
- Padding positions are **included**, because no mask is applied.

### Why multiply by $T^2$

The gradient of the KL term with respect to a student logit is

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_{t,i}} \propto \frac{1}{T} \left( p_{t,i} - q_{t,i} \right)
$$

The $\frac{1}{T}$ comes from the chain rule through $z / T$. For large $T$, $e^{x} \approx 1 + x$, so the softmax is approximately linear:

$$
p_{t,i} \approx \frac{1 + z_{t,i}/T}{V}
\quad\Longrightarrow\quad
p_{t,i} - q_{t,i} \approx \frac{z_{t,i} - v_{t,i}}{V \, T}
$$

Combining the two:

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_{t,i}} \approx \frac{z_{t,i} - v_{t,i}}{V \, T^2}
$$

The gradient shrinks like $\frac{1}{T^2}$. Multiplying the loss by $T^2$ cancels this, so the distillation gradient keeps roughly the same magnitude for any $T$ and $\alpha$ keeps its meaning. This follows [Hinton et al., 2015](https://arxiv.org/abs/1503.02531). The approximation is best when $T$ is large relative to the logit differences, so the correction is not exact at small $T$.

## Text generation

`infer.py` generates one token at a time. At each step, with student logits $z$ for the next token:

**1. Repetition penalty** ($\rho = 1.2$): for every token $i$ already in the text,

$$
z_i \leftarrow
\begin{cases}
z_i / \rho & z_i > 0 \\
z_i \cdot \rho & z_i \le 0
\end{cases}
$$

**2. Temperature** ($\tau = 0.7$, separate from the training $T$): $\tau < 1$ sharpens the distribution.

$$
P(i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

**3. Top-p (nucleus) filtering** ($p = 0.9$): sort tokens by probability and keep the smallest set $S$ such that

$$
\sum_{i \in S} P(i) \ge 0.9
$$

then renormalize over $S$ and sample the next token from it. With $\tau = 0$ the script instead uses greedy decoding, $\arg\max_i z_i$.

## Setup

```bash
pip install -r requirements.txt
```

Models and the dataset are downloaded from Hugging Face on first run. The dataset is streamed, so only the samples used are fetched.

## Usage

### Train

```bash
python3 main.py
```

- Streams 2,000 documents from `fineweb-edu` (`sample-10BT`): 1,800 for training, 200 for evaluation.
- Trains for 1 epoch, evaluating every 50 steps.
- Saves the distilled student to `./final_distilled_text_model`.

### Generate text

```bash
python3 infer.py
```

Loads `./final_distilled_text_model` and prints a completion for a fixed prompt. Edit the constants at the top of `infer.py` to change it:

| Constant | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `./final_distilled_text_model` | Model to load. Set to `EleutherAI/pythia-70m` to compare against the undistilled student. |
| `PROMPT` | `"The water cycle is"` | Text to continue. |
| `MAX_NEW_TOKENS` | `100` | Length of the completion. |
| `TEMPERATURE` | `0.7` | Sampling randomness; `0` for greedy decoding. |

## GPU selection

Both scripts look up the GPU whose name contains `p100` via `nvidia-smi` and hide every other GPU (`CUDA_VISIBLE_DEVICES`) before torch starts. On a machine with more than one GPU, this keeps the Trainer from splitting work across cards. If no P100 is found, the scripts fall back to the default CUDA devices.

To target a different card, change the `select_gpu_by_name("p100")` call in each script.

## Hardware notes

The P100-PCIe is passively cooled and depends on server chassis airflow. In a desktop it can overheat or overload the power supply under sustained load. The training settings are kept small for this reason (batch size 4, 128-token sequences, no dataloader workers). Additionally:

- **Cap the power draw** (resets on reboot; GPU index from `nvidia-smi`):
  ```bash
  sudo nvidia-smi -i 1 -pm 1 && sudo nvidia-smi -i 1 -pl 125
  ```
- **Monitor temperature** during training:
  ```bash
  watch -n 2 nvidia-smi --query-gpu=index,temperature.gpu,power.draw --format=csv
  ```
  Stop if the P100 goes above ~85 °C and add a fan directed at the card.

## Training configuration

| Setting | Value |
|---|---|
| Teacher / student | `pythia-160m` / `pythia-70m` |
| Max sequence length | 128 tokens |
| Batch size | 4 |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Precision | fp16 mixed precision (fp32 weights) |
| `alpha` / `temperature` | 0.5 / 2.0 |

## Project structure

```
main.py           # Distillation training
infer.py          # Text generation with the distilled model
requirements.txt  # Python dependencies
```
