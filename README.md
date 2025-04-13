# Prerequisites
Python 3.11.11

# Installation
1. Clone the repo or unzip the `counterfactual_explainers-main.zip`.
   ```sh
   git clone https://github.com/Hariss-Gills/counterfactual_explainers.git
   ```
2. Create virtual environments for each counterfactual method.
   ```sh
   python -m venv ~/.python-venvs/counterfactual-explainers-dice
   python -m venv ~/.python-venvs/counterfactual-explainers-aide
   ```
3. Install the packages required for each method.
   ```sh
   cd counterfactual_explainers
   source ~/.python-venvs/counterfactual-explainers-dice/bin/activate
   pip install .
   pip install -r requirements-dice.txt
   deactivate
   source ~/.python-venvs/counterfactual-explainers-aide/bin/activate
   pip install .
   pip install -r requirements-aide.txt
   deactivate
   ```

# Usage
Run the top level modules .
   ```sh
   source ~/.python-venvs/counterfactual-explainers-dice/bin/activate
   python counterfactual_explainers/dataset_stats.py
   python counterfactual_explainers/train_models.py
   python counterfactual_explainers/gen_cfs_dice.py
   deactivate
   source ~/.python-venvs/counterfactual-explainers-aide/bin/activate
   python counterfactual_explainers/gen_cfs_aide.py
   python counterfactual_explainers/calculate_metrics.py
   python counterfactual_explainers/plot_and_stats.py
   deactivate
   ```

# Notes
- Need to downgrade from python 3.13.1 to 3.12.8 due to tensorflow
- Need to downgrade to tensorflow==2.13.0 since dice_ml==0.11 used that version. I wonder If downgrading sklearn as well might fix the desearlizing issue between using pipeline vs raw model. Due to this, I had to downgrade python to 3.11.11 
- Since [`ClassifierMixin`](https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/base.py#L540) does not have a superclass, I need to use `Tags()` class to instatiate some tags instead of downgrading scikit-learn.
