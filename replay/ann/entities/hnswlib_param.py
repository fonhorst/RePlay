from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class HnswlibParam:
    """
    Parameters for hnswlib methods.

    For example,

    >>> HnswlibParam(space="ip",\
                     M=100,\
                     efC=200,\
                     post=0,\
                     efS=2000,\
                     build_index_on="driver"\
        )
    or
    >>> HnswlibParam(space="ip",\
                     M=100,\
                     efC=200,\
                     post=0,\
                     efS=2000,\
                     build_index_on="executor"\
                     index_path="/tmp/hnswlib_index"\
        )

    The "space" parameter described on the page https://github.com/nmslib/hnswlib/blob/master/README.md#supported-distances
    Parameters "M", "efS" and "efC" are described at https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

    The reasonable range of values for M parameter is 5-100,
    for efC and eFS is 100-2000.
    Increasing these values improves the prediction quality
    but increases index_time and inference_time too.

     We recommend using these settings:

    - M=16, efC=200 and efS=200 for simple datasets like MovieLens.
    - M=50, efC=1000 and efS=1000 for average quality with an average prediction time.
    - M=75, efC=2000 and efS=2000 for the highest quality with a long prediction time.

    note: choosing these parameters depends on the dataset
    and quality/time tradeoff.

    note: while reducing parameter values the highest range metrics
    like Metric@1000 suffer first.

    note: even in a case with a long training time,
    profit from ann could be obtained while inference will be used multiple times.
    """

    space: Literal["l2", "ip", "cosine"] = "ip"
    M: int = 200
    efC: int = 20000
    post: int = 0
    efS: Optional[int] = None
    build_index_on: Literal["driver", "executor"] = "driver"
    index_path: Optional[str] = None
    # Dimension of vectors in index
    dim: int = field(default=None, init=False)
    # Max number of elements that will be stored in the index
    max_elements: int = field(default=None, init=False)

    def __post_init__(self):
        if self.build_index_on == "executor":
            assert (
                self.index_path
            ), 'if build_index_on == "executor" then index_path must be set!'
