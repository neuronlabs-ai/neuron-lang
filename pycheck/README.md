# ⚡ PyCheck: Catch Temporal Lookahead Bias in 15ms

`pycheck` is an ultra-fast static AST analyzer for Python that catches hidden **temporal lookahead bias** (future data leakage), causal confusion, and unguarded uncertainty in machine learning and algorithmic trading scripts before execution.

## 🚀 Quickstart

```bash
pip install pycheck
pycheck my_trading_strategy.py
```

## 🔍 What It Catches

- **`TemporalLeak`**: Catches `df.shift(-1)` negative shift lookahead target alignment, `train_test_split` shuffling on time series, or rolling statistics calculated before train/test splits.
- **`CausalConfusion`**: Identifies `.corr()` correlations mistakenly used for treatment or trading decision logic.
- **`UncertaintyIgnored`**: Detects point-estimate `.predict()` calls without confidence interval or probability evaluation.

## 💡 Beyond Python: NEURON Language

For 100% compile-time temporal proofs, type-checked Causal Do-calculus, and 21ms native C-ABI execution speed, check out [NEURON](https://github.com/your-username/neuron-lang).

## 📄 License

MIT License
