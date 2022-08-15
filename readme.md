# Adversarial Training on CIFAR

This repo implements two adversarial training algorithms:

**PGD-AT**: [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)

**TRADES**: [Theoretically principled trade-off between robustness and accuracy](https://proceedings.mlr.press/v97/zhang19p.html)

## Run
To run the model, simply run the following command:

```
python algorithms/trainer_trades.py
```
with settings specified in trainer_at.py or trainer_trades.py.

The checkpoints are also given in trained_model with best model named as `best.pth`.

## Settings

In PGD-AT, the initial point is randomly generated in the tolerable attack range. In TRADES, the initial point is randomly generated in a small range according to the original paper.

We train each robust model with 150 epochs, though in most of the case, 100 epochs are enough to have a good result.

The robust models are evaluated using [foolbox](https://github.com/bethgelab/foolbox)

## Accuracy

| Algorithm | beta | Clean Acc   | PGD-20 Acc  |
| --------- | ---- | ----------- | ----------- |
| TRADES    | 5.0  | 83.52%      | **51.79%** |
| TRADES    | 1.0  | **87.18%** | 45.49%      |
| PGD-AT    | 0.5  | 85.39%      | 45.49%      |
| PGD-AT    | 1.0  | 84.07%      | 49.63%      |

## Training Time

| Algorithm | Devices   | Time         |
| --------- | --------- | ------------ |
| TRADES    | RTX A6000 | 6.2min/epoch |
| AT-PGD    | GF 3090   | 3.2min/epoch |
