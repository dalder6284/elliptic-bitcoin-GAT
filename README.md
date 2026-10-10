# **CPSC 483: Bitcoin Fraud Detection using Graph Attention Networks**
An implementation for detecting illicit transactions in cryptocurrency networks, using the Elliptic Bitcoin dataset as a benchmark. The project combines the power of Graph Attention Networks (GATs) with a stabilized self-training framework, exploring their impact on performance on highly imbalanced datasets. For CPSC 483.

## Usage

The corrected experiment is in `AldereteDiego_BitcoinGAT.ipynb`. It expects Python 3.10 or later and the Elliptic CSV files under `datasets/elliptic_bitcoin_dataset/`.

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead of `source .venv/bin/activate`.

Open `AldereteDiego_BitcoinGAT.ipynb`, select the `.venv` Python kernel, and use **Run All**. It trains each configuration with the same ten seeds (`42`–`51`), for a maximum of 1,000 epochs per run:

- One attention head with 99 hidden channels.
- Three attention heads with 33 channels per head.
- Three attention heads with 33 channels per head and corrected stabilized self-training.

The split is chronological: timesteps 1–30 train the weights, 31–34 select the checkpoint and illicit-class threshold, and 35–49 are used once for final testing. Training stops after 100 epochs without an improvement in pooled validation illicit-class F1.

Testing reports loss, accuracy, both classes' precision/recall/F1, illicit-class PR-AUC and average precision, and confusion counts. It produces per-timestep files plus pooled summaries for the full test period, pre-shutdown timesteps 35–41, shutdown timestep 42, and post-shutdown timesteps 43–49. The final summary reports mean and sample standard deviation across the ten seeds.

New artifacts use the `balanced_` prefix and are written to `models/`, `training_results/`, and `testing_results/`. Results from the earlier experiment are retained under `historical/`. In particular:

- `training_results/balanced_training_manifest.csv` records each validation-selected checkpoint.
- `testing_results/balanced_test_timestep_metrics.csv` combines all per-timestep results.
- `testing_results/balanced_test_seed_metrics.csv` contains pooled metrics for every seed and test region.
- `testing_results/balanced_test_summary.csv` contains the final mean ± standard deviation results.
- `testing_results/balanced_illicit_f1_by_timestep.png` plots the shutdown-era F1 change with seed variability.

The complete experiment runs 30 training jobs sequentially: three configurations across ten seeds (`42`–`51`), so all final checkpoints and result files come from one consistent execution.

If the dataset is missing, download it from the [verified Kaggle page](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set/data) and preserve the directory structure above.

### Contact
Let me know if you have any questions at diegoalderetellaca@gmail.com
