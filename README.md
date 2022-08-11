# RePlay

RePlay is a library providing tools for all stages of creating a recommendation system, from data preprocessing to model evaluation and comparison.

RePlay uses PySpark to handle big data.

You can

- Filter and split data
- Train models
- Optimize hyper parameters
- Evaluate predictions with metrics
- Combine predictions from different models
- Create a two-level model


- [Documentation](https://sb-ai-lab.github.io/RePlay/)
- [Framework architecture](https://miro.com/app/board/uXjVOhTSHK0=/?share_link_id=748466292621)


<a name="toc"></a>
# Table of Contents

* [Installation](#installation)
* [Quickstart](#quickstart)
* [Resources](#examples)
* [Contributing to RePlay](#contributing)


<a name="installation"></a>
## Installation

Use Linux machine with Python 3.7+, Java 8+ and C++ compiler.

```bash
pip install replay-rec
```

To get the latest development version or RePlay, [install it from the GitHab repository](https://sb-ai-lab.github.io/RePlay/pages/installation.html#development).
It is preferable to use a virtual environment for your installation.

If you encounter an error during RePlay installation, check the [troubleshooting](https://sb-ai-lab.github.io/RePlay/pages/installation.html#troubleshooting) guide.


<a name="quickstart"></a>
## Quickstart

```python
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator, Indexer
from replay.metrics import HitRate, NDCG
from replay.models import KNN
from replay.session_handler import State
from replay.splitters import UserSplitter

spark = State().session

ml_1m = MovieLens("1m")

# data preprocessing
preparator = DataPreparator()
log = preparator.transform(
    columns_mapping={
        'user_id': 'user_id',
        'item_id': 'item_id',
        'relevance': 'rating',
        'timestamp': 'timestamp'
    }, 
    data=ml_1m.ratings
)
indexer = Indexer(user_col='user_id', item_col='item_id')
indexer.fit(users=log.select('user_id'), items=log.select('item_id'))
log_replay = indexer.transform(df=log)

# data splitting
user_splitter = UserSplitter(
    item_test_size=10,
    user_test_size=500,
    drop_cold_items=True,
    drop_cold_users=True,
    shuffle=True,
    seed=42,
)
train, test = user_splitter.split(log_replay)

# model training
model = KNN()
model.fit(train)

# model inference
recs = model.predict(
    log=train,
    k=K,
    users=test.select('user_idx').distinct(),
    filter_seen_items=True,
)

# model evaluation
metrics = Experiment(test,  {NDCG(): K, HitRate(): K})
metrics.add_result("knn", recs)
```

<a name="examples"></a>
## Resources

### Examples in google colab
1. [01_replay_basics.ipynb](https://colab.research.google.com/github/sb-ai-lab/RePlay/blob/main/experiments/01_replay_basics.ipynb)
2. [02_models_comparison.ipynb](https://colab.research.google.com/github/sb-ai-lab/RePlay/blob/main/experiments/02_models_comparison.ipynb)
3. [03_features_preprocessing_and_lightFM.ipynb](https://colab.research.google.com/github/sb-ai-lab/RePlay/blob/main/experiments/03_features_preprocessing_and_lightFM.ipynb)


### Videos and papers
* **Video guides**:
	- (Russian) [AI Journey 2021](https://www.youtube.com/watch?v=M9XqEJb2Ncc)

* **Papers**:
	- Yan-Martin Tamm, Rinchin Damdinov, Alexey Vasilev [Quality Metrics in Recommender Systems: Do We Calculate Metrics Consistently?](https://dl.acm.org/doi/10.1145/3460231.3478848)

<a name="contributing"></a>
## Contributing to RePlay

For more details visit [development section in docs](https://sb-ai-lab.github.io/RePlay/pages/installation.html#development)
