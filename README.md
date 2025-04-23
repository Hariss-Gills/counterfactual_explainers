# 🎲 Evaluating Contrastive Explanations: Rolling the DiCE with AIDE

This project addresses the critical need for explainability in opaque machine learning models by evaluating and comparing contrastive counterfactual explanation methods. Specifically, it investigates the performance of Diverse Counterfactual Explanations (DiCE), its genetic variant based on GeCo, and the immune-inspired Artificial Immune Diverse Explanations (AIDE) algorithm. The study employs a mixed-methods approach, combining quantitative evaluation across established metrics (including Size, Dissimilarity, Actionability, Diversity, and Runtime) with qualitative analysis using parallel coordinate plots on four standard benchmark datasets (adult, fico, compas, german\_credit). Quantitative results indicate that no single method universally outperforms others; which is in agreement with what has been reported in the literature. DiCE typically generates explanations closest to the original instance, while AIDE excels in producing diverse and actionable explanations, albeit at a higher computational cost. The genetic DiCE variant offers the fastest performance but can struggle with consistency. Using principles in Exploratory Data Analysis, parallel coordinates plots were used to visualize the counterfactuals. By going deeper, this analysis highlights differences in how methods handle feature types and how the counterfactuals align with dataset. Hence, a novel "Alignment" metric is proposed to assess the realism of generated counterfactuals relative to the data distribution of the counterfactual class, capturing the "concentration" of values. The evaluation study with the new metric concludes that the optimal choice of counterfactual explainer depends on the specific dataset. Additionally, implementations might consider employing a combination of counterfactual methods within an XAI system. Knowing that good explanations are also selective, the system should select the explanation based on the relative importance of different evaluation metrics from the end-user's perspective.

---
## 🧩 Software Dependencies

This project uses Python=3.11.11. Due to specific library requirements, different versions of TensorFlow are needed for different parts of the project:
* **DiCE**: tensorflow==2.13.0
* **AIDE**: tensorflow==2.18.0

Following the installation instructions below will ensure all software dependencies are installed appropriately into separate environments.

## ⚙️ Installation

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

## ⚒️ Usage
Run the top level modules.
   ```sh
   source ~/.python-venvs/counterfactual-explainers-dice/bin/activate
   python counterfactual_explainers/dataset_stats.py
   python counterfactual_explainers/train_models.py # -e argument to specify explainer type
   python counterfactual_explainers/gen_cfs_dice.py
   deactivate
   source ~/.python-venvs/counterfactual-explainers-aide/bin/activate
   python counterfactual_explainers/gen_cfs_aide.py
   python counterfactual_explainers/calculate_metrics.py
   python counterfactual_explainers/plot_and_stats.py
   deactivate
   ```

## 📝 Notes
- Need to downgrade from python 3.13.1 to 3.12.8 due to tensorflow
- Need to downgrade to tensorflow==2.13.0 since dice_ml==0.11 used that version. I wonder If downgrading sklearn as well might fix the desearlizing issue between using pipeline vs raw model. Due to this, I had to downgrade python to 3.11.11 
- Since [`ClassifierMixin`](https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/base.py#L540) does not have a superclass, I need to use `Tags()` class to instatiate some tags instead of downgrading scikit-learn.
