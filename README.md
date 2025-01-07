### Data sources

- [adult](https://archive.ics.uci.edu/dataset/2/adult)
- [compas](https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv)
- [fico](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc)
- [german](https://www.kaggle.com/datasets/renaldydermawan25/credit-data)

# Notes
Needed to downgrade from python 3.13.1 to 3.12.8 due to tensorflow

Since [`ClassifierMixin`](https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/base.py#L540) does not have a superclass, I need to use `Tags()` class to instatiate some tags
instead of downgrading scikit-learn.

