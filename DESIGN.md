# Design Notes: Bridge Bidding via Self-Play Reinforcement Learning

## 1  Motivation and Scope

The goal is to build a bridge bidding agent that discovers bidding conventions without any prior knowledge of established systems (SAYC, 2/1, Precision, etc.). Rather than imitating human expert bids, the agent is rewarded purely for reaching good contracts as measured against the double-dummy par score. If a novel or unconventional sequence reliably reaches the best contract, the agent is free to adopt it.

The current scope is **Phase 1: uncontested bidding**. East–West always pass; only North and South bid freely. This isolates the cooperative communication problem — can two agents, sharing a single network, develop a common bidding language from scratch? Phase 2 will introduce adversarial East–West agents.

---

## 2  Environment

### 2.1  Deal representation

Each of the 52 cards is assigned a fixed index (suit × 13 + rank). A player's hand is encoded as a **52-dimensional binary vector** with a 1 at each card the player holds. This representation is permutation-invariant within a suit (there is no ordering imposed on cards within the hand), dense enough to be fed directly into an MLP, and lossless — the full hand is recoverable from the vector.

### 2.2  Auction state machine

The bid space has **38 actions**:

| Index | Meaning |
|-------|---------|
| 0 | Pass |
| 1–35 | Contract bids 1♣ through 7NT, in order (level−1)×5 + denom + 1 |
| 36 | Double |
| 37 | Redouble |

The auction state machine tracks the last non-pass bid, the doubling status, and the number of consecutive trailing passes. The auction ends when three consecutive passes follow the first real bid, or four consecutive passes occur from the opening (a passed-out board). Valid bids at each step are enforced: a Double requires the last contract bid to have come from an opponent, a Redouble requires the last action to have been a Double from an opponent, and any contract bid must be strictly higher than the last.

Declarer is defined as the first player on the declaring side who bid the contract's denomination — the standard bridge rule.

### 2.3  Double-dummy oracle and scoring

The reward signal comes from comparing the contract reached by the agents to the **par contract** — the theoretically optimal result under double-dummy play. Par is computed by the [endplay](https://github.com/dominicprice/endplay) library, which wraps Bo Haglund's DDS (double dummy solver), a widely used open-source C++ implementation.

**IMP scoring** converts the point difference between the achieved score and the par score into the interval [−24, +24] using the standard duplicate bridge IMP table. A positive reward means the agents exceeded par; zero means they reached par exactly; negative means they left value on the table (or bid too high and went down).

---

## 3  Counterfactual Reward Estimation

A single East–West layout introduces substantial variance into the reward. If East–West happen to hold cards that make North–South's best strain easy, the reward is positive even for a mediocre bid sequence. Conversely, an unlucky EW layout can punish a good sequence.

To reduce this variance, the reward for a given North–South auction is estimated as the **expectation over K independently re-sampled EW completions**:

$$r = \frac{1}{K} \sum_{k=1}^{K} \mathrm{IMP}\!\left(\mathrm{score}(A, \text{EW}_k) - \mathrm{par}(\text{NS}, \text{EW}_k)\right)$$

where $A$ is the auction sequence, $\text{EW}_k$ is the $k$-th random EW completion holding NS fixed, and par is also computed per EW completion so the baseline is consistent with each specific layout. The NS hands are fixed (they are what drove the bidding), but the EW cards are re-shuffled $K$ times and DDS is queried for each combination. This is an unbiased estimator of the expected IMP reward conditioned on the NS hands.

---

## 4  Pre-Computed Dataset

Running the DDS solver at training time is the dominant cost (~14 boards/second on a laptop). To decouple data generation from training, deals are pre-computed offline:

- **`ns_hands.npy`** — shape $(N, 2, 52)$ uint8: North and South hand vectors for $N$ deals.
- **`dds_tables.npy`** — shape $(N, K, 5, 4)$ uint8: double-dummy trick counts for $K$ EW completions of each deal, across all 5 denominations and 4 declaring directions.

Both files are loaded as numpy memory-maps so only accessed pages enter RAM. At training time, reward computation is pure numpy (lookups into the pre-computed tables), with no DDS calls. This yields roughly a 200× speedup over the live-DDS path, making GPU-accelerated training practical.

---

## 5  Network Architecture

North and South share a **single set of weights** (`BiddingNet`). The network receives the agent's own hand and the full auction history, and outputs a policy (action distribution) and a value estimate. The seat (N or S) is communicated to the shared network via a **direction token** prepended to the auction sequence, so the model can condition its output on which position it is bidding from.

```
hand  (52,)            →  Hand encoder MLP  →  hand_emb  (H,)
                                                              ↘
auction  (T,)  →  Embedding  →  (T, E)  →  LSTM  →  lstm_emb  (H,)
                                                              ↙
                                              concat  (2H,)
                                                  ↓
                                       Policy head MLP  →  logits  (38,)
                                       Value head MLP   →  value   (1,)
```

**Hand encoder**: A multi-layer perceptron with ReLU activations mapping the 52-dimensional binary hand vector to a $H$-dimensional embedding.

**Auction encoder**: Each bid token (including the leading direction token) is looked up in a learned embedding table of dimension $E$. The sequence is then processed by a stacked LSTM. Sequences are packed before being fed to the LSTM so no computation is wasted on padding tokens and gradients do not flow through them. The final hidden state of the top LSTM layer is used as the auction embedding.

**Direction token**: The agent's seat (0–3) is encoded as a special vocabulary token appended at offset $\text{NUM\_BIDS}$, so it sits outside the bid range and is prepended to the auction sequence before LSTM encoding. This lets a single shared network behave differently depending on whether it is North or South.

**Output heads**: The combined $(2H)$-dimensional vector feeds two independent MLPs — a policy head producing unnormalized logits over 38 actions, and a value head producing a scalar estimate of the expected return. Illegal actions are masked to $-10^9$ before the log-softmax, which prevents probability mass on impossible bids while avoiding the NaN gradients that arise from exact $-\infty$ masking in some autograd implementations.

**Default hyperparameters**:

| Parameter | Default |
|-----------|---------|
| Hidden width $H$ | 128 |
| Embedding dim $E$ | 32 |
| MLP hidden layers | 1 |
| LSTM layers | 1 |
| ~Parameters | 124 k |

---

## 6  Training Algorithm

### 6.1  PPO

The agent is trained with **Proximal Policy Optimization (PPO)** with clipping. PPO is well suited here because:

- It is stable under the noisy, sparse reward signal (a single IMP value per auction).
- The clipping mechanism prevents destructive large updates when the policy changes rapidly early in training.
- It naturally accommodates the cooperative N–S setup: both agents share weights and both receive the same terminal reward, so the gradient signal flows symmetrically through both players' transitions.

The PPO objective is:

$$\mathcal{L}_\pi = -\mathbb{E}\left[\min\!\left(r_t \hat{A}_t,\ \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

with $r_t = \pi_\theta(a_t|s_t) / \pi_{\theta_\mathrm{old}}(a_t|s_t)$ and $\varepsilon = 0.2$.

The full loss adds a value regression term and an entropy bonus:

$$\mathcal{L} = \mathcal{L}_\pi + c_v \mathcal{L}_v - c_H H(\pi)$$

with $c_v = 0.5$, $c_H = 0.01$.

### 6.2  Return computation

Each auction produces a single terminal reward (the mean IMP over $K$ EW completions). Intermediate rewards are zero. Returns are computed with $\gamma = 1.0$ — appropriate for short episodes with a single terminal payoff — by a simple backward pass over each episode's transitions.

Advantages are estimated as $\hat{A}_t = G_t - V(s_t)$ and normalized to zero mean and unit variance within each batch before the PPO update.

### 6.3  Vectorized rollout collection

To fully utilize a GPU with a large network, rollout collection is vectorized across all $N$ episodes in a batch. All $N$ auctions advance in lockstep: at each step, every active auction takes exactly one action. At North/South steps, the observations for all active N/S auctions in that step are stacked into a single batch tensor and processed by a single forward pass. At East/West steps, forced passes are applied with no network call. This reduces the number of forward passes per batch from $O(N \times \text{auction length})$ sequential size-1 passes to $O(\text{auction length})$ batched passes of size $\approx N/4$, which is far more efficient on a GPU.

### 6.4  Learning rate schedule

The learning rate decays linearly from $3 \times 10^{-4}$ to $1 \times 10^{-4}$ over the first two-thirds of training, then holds constant. Updates use Adam with gradient clipping at norm 0.5.

---

## 7  What the Reward Incentivises

The IMP table compresses the raw point difference logarithmically. This has a useful training effect: the gradient signal for catastrophically bad contracts (large negative IMPs) is not disproportionately large compared to incrementally good ones. The agent is rewarded for:

1. **Bidding game when game makes** — the IMP swing for missing a vulnerable game is 10–12 IMPs, a strong positive incentive.
2. **Bidding the right strain** — a major-suit fit vs a minor-suit contract can be several hundred points, ~6 IMPs.
3. **Not overbidding** — going down in a contract below par costs IMPs roughly proportionally to undertricks.
4. **Slam bidding** — small slams score an additional bonus (500/750 non-vul/vul), large slams more, making them worth 11–13 extra IMPs over game.

Because the reward is always measured relative to par, the agent has no incentive to settle for a mediocre positive score when a better contract is available.

---

## 8  Phase 2: Competitive Bidding

The architecture is designed to extend cleanly to four-player competitive play:

- East and West receive a separate (or shared) `BiddingNet`, initialized from the trained N–S weights.
- The `uncontested=True` flag in `AuctionState` is removed; Double and Redouble become live actions for E/W.
- The EW reward is simply $-r_\text{NS}$ (zero-sum within each board).
- Competitive auctions can be trained adversarially: freeze NS, improve EW; freeze EW, improve NS; repeat.

The bid space (38 actions) and auction state machine already handle the full competitive case without modification.
