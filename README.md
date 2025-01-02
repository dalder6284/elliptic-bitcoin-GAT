# **CPSC 483: Bitcoin Fraud Detection using Graph Attention Networks**
An implementation for detecting illicit transactions in cryptocurrency networks, using the Elliptic Bitcoin dataset as a benchmark. The project combines the power of Graph Attention Networks (GATs) with a stabilized self-training framework, exploring their impact on performance on highly imbalanced datasets. For CPSC 483.

## Usage
There is a Jupyter Notebook (`.ipynb`) file titled `AldereteDiego_BitcoinGAT.ipynb` in the main branch. The imports assume you have **PyTorch, Numpy, and Pandas** installead but create a way to install `torch_scatter`, `torch_sparse`, and `torch_geometric` given that you have the latest version of PyTorch installed. Make sure to download the data set from this [Kaggle page](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set/data) and place it in the `datasets` folder. Other than that, the instructions should be pretty straightforward within the Jupyter notebook. 

### Contact
Let me know if you have any questions at diegoalderetellaca@gmail.com
