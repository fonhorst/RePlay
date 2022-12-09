import dataclasses
import logging.config
import os
import shutil
import uuid
from typing import cast

import pytest
from pyspark.sql import SparkSession
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT

import replay
from conftest import phase_report_key
from experiments.two_stage_scenarios import dataset_splitting, first_level_fitting, ArtifactPaths, \
    second_level_fitting, init_refitable_two_stage_scenario, \
    combine_train_predicts_for_second_level, RefitableTwoStageScenario, _init_spark_session, EmptyRecommender, \
    presplit_data, fit_predict_first_level_model, PartialTwoStageScenario, dense_hnsw_params
from replay.data_preparator import ToNumericFeatureTransformer
from replay.history_based_fp import EmptyFeatureProcessor, LogStatFeaturesProcessor, ConditionalPopularityProcessor, \
    HistoryBasedFeaturesProcessor
from replay.model_handler import load
from replay.utils import save_transformer, load_transformer


logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def spark_sess() -> SparkSession:
    with _init_spark_session(cpu=4, memory=4) as spark_s:
        os.environ["INIT_SPARK_SESSION_STOP_SESSION"] = "0"
        yield spark_s
        del os.environ["INIT_SPARK_SESSION_STOP_SESSION"]


@pytest.fixture
def resource_path() -> str:
    return "/opt/data/resources"


@pytest.fixture(scope='function')
def ctx(request, resource_path: str, artifacts: ArtifactPaths):
    # copy the context of dependencies
    context_name = request.param
    required_resource_path = os.path.join(resource_path, context_name)
    assert os.path.exists(required_resource_path)
    shutil.rmtree(artifacts.base_path, ignore_errors=True)
    shutil.copytree(required_resource_path, artifacts.base_path)


@pytest.fixture(scope="function")
def artifacts(request, resource_path: str) -> ArtifactPaths:
    path = "/opt/experiments/test_exp"
    data_base_path = "/opt/data/"

    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

    yield ArtifactPaths(
        base_path=path,
        log_path=os.path.join(data_base_path, "ml100k_ratings.csv"),
        user_features_path=os.path.join(data_base_path, "ml100k_users.csv"),
        item_features_path=os.path.join(data_base_path, "ml100k_items.csv")
    )

    report = request.node.stash[phase_report_key]
    if report["setup"].failed:
        print("setting up a test failed or skipped", request.node.nodeid)
    elif ("call" not in report) or report["call"].failed:
        print("executing test failed or skipped", request.node.nodeid)
    else:
        test_name = request.node.name
        # process cases like this "test_init_refitable_two_stage_scenario[test_data_splitting__out]"
        test_name = test_name[:test_name.find('[')] if test_name.endswith(']') else test_name
        resource_test_path = os.path.join(resource_path, f"{test_name}__out")
        if os.path.exists(resource_test_path):
            shutil.rmtree(resource_test_path)

        os.makedirs(resource_path, exist_ok=True)
        shutil.copytree(path, resource_test_path)
        print(f"Copied the exp dir state ({path}) to the resource folder "
              f"of this test ({resource_test_path}) for next tests")

    # shutil.rmtree(path, ignore_errors=True)


def test_data_splitting(spark_sess: SparkSession, artifacts: ArtifactPaths):
    dataset_splitting.function(
        artifacts,
        partitions_num=4,
        dataset_name="ml100k"
    )

    assert os.path.exists(artifacts.base_path)
    assert os.path.exists(artifacts.train_path)
    assert os.path.exists(artifacts.test_path)

    # TODO: they are not empty, no crossing by ids


@pytest.mark.parametrize('ctx', ['test_data_splitting__out'], indirect=True)
def test_init_refitable_two_stage_scenario(spark_sess: SparkSession, artifacts: ArtifactPaths, resource_path: str, ctx):
    init_refitable_two_stage_scenario.function(
        artifacts
    )

    assert os.path.exists(artifacts.two_stage_scenario_path)
    assert os.path.exists(os.path.join(artifacts.base_path, "first_level_train.parquet"))
    assert os.path.exists(os.path.join(artifacts.base_path, "second_level_positive.parquet"))
    assert os.path.exists(os.path.join(artifacts.base_path, "first_level_candidates.parquet"))

    setattr(replay.model_handler, 'EmptyRecommender', EmptyRecommender)
    setattr(replay.model_handler, 'RefitableTwoStageScenario', RefitableTwoStageScenario)
    scenario = cast(RefitableTwoStageScenario, load(artifacts.two_stage_scenario_path))

    assert scenario._are_candidates_dumped
    assert scenario._are_split_data_dumped
    assert scenario.features_processor.fitted
    assert scenario.first_level_item_features_transformer.fitted
    assert scenario.first_level_user_features_transformer.fitted

    result = scenario.predict(
        log=artifacts.train, user_features=artifacts.user_features, item_features=artifacts.item_features,k=20
    )
    assert result.count() > 0


@pytest.mark.parametrize('ctx', ['test_init_refitable_two_stage_scenario__out'], indirect=True)
def test_first_level_fitting(spark_sess: SparkSession, artifacts: ArtifactPaths, ctx):
    # alternative
    model_class_name = "replay.models.knn.ItemKNN"
    model_kwargs = {"num_neighbours": 10}

    # model_class_name, model_kwargs = "replay.models.ucb.UCB", {"seed": 42}
    # model_class_name, model_kwargs = "replay.models.word2vec.Word2VecRec", {"rank": 10, "seed": 42}

    # alternative
    # model_class_name = "replay.models.als.ALSWrap"
    # model_kwargs={"rank": 10}

    first_level_fitting(artifacts, model_class_name, model_kwargs, k=10)

    assert os.path.exists(artifacts.model_path(model_class_name))
    assert os.path.exists(artifacts.partial_train_path(model_class_name))
    assert os.path.exists(artifacts.partial_predicts_path(model_class_name))

    # second model (use)
    next_artifacts = dataclasses.replace(artifacts, uid=str(uuid.uuid4()).replace('-', ''))
    next_model_class_name = "replay.models.als.ALSWrap"
    next_model_kwargs = {"rank": 10}

    first_level_fitting(next_artifacts, next_model_class_name, next_model_kwargs, k=10)

    assert os.path.exists(next_artifacts.model_path(next_model_class_name))
    assert os.path.exists(next_artifacts.partial_train_path(next_model_class_name))
    assert os.path.exists(next_artifacts.partial_predicts_path(next_model_class_name))

    # TODO: restore this checking later
    # spark = _get_spark_session()
    #
    # ptrain_1_df = spark.read.parquet(artifacts.partial_train_path(model_class_name))
    # ppreds_1_df = spark.read.parquet(artifacts.predictions_path(model_class_name))
    # ptrain_2_df = spark.read.parquet(new_artifacts.partial_train_path(next_model_class_name))
    # ppreds_2_df = spark.read.parquet(new_artifacts.predictions_path(next_model_class_name))
    #
    # assert ptrain_1_df.count() == ptrain_2_df.count()
    # assert ppreds_1_df.count() == ppreds_2_df.count()
    #
    # pt_uniques_1_df = ptrain_1_df.select("user_idx", "item_idx").distinct()
    # pt_uniques_2_df = ptrain_2_df.select("user_idx", "item_idx").distinct()
    #
    # assert ptrain_1_df.count() == pt_uniques_1_df.join(pt_uniques_2_df, on=["user_idx", "item_idx"]).count()
    #
    # spark.stop()
    # TODO: both datasets can be combined


@pytest.mark.parametrize('ctx', ['test_first_level_fitting__out'], indirect=True)
def test_combine_datasets(spark_sess: SparkSession, artifacts: ArtifactPaths, ctx):
    combine_train_predicts_for_second_level.function(artifacts)

    assert os.path.exists(artifacts.full_second_level_train_path)
    assert os.path.exists(artifacts.full_second_level_predicts_path)


@pytest.mark.parametrize('ctx', ['test_combine_datasets__out'], indirect=True)
def test_second_level_fitting(spark_sess: SparkSession, artifacts: ArtifactPaths, ctx):
    model_name = "test_lama_model"

    second_level_fitting(
        artifacts=artifacts,
        model_name=model_name,
        k=10,
        second_model_type="slama",
        # second_model_type="lama",
        second_model_params={
            "general_params": {"use_algos": [["lgb"]]},
            # "lgb_params": {
            #     'default_params': {'numIteration': 10}
            # },
            "reader_params": {"cv": 5, "advanced_roles": False}
        },
        second_model_config_path=None
    )

    # TODO: restore this test later
    assert os.path.exists(artifacts.second_level_predicts_path(model_name))
    assert os.path.exists(artifacts.second_level_model_path(model_name))


@pytest.mark.parametrize('ctx', ['test_data_splitting__out'], indirect=True)
@pytest.mark.parametrize('transformer', [
    EmptyFeatureProcessor(),
    LogStatFeaturesProcessor(),
    ConditionalPopularityProcessor(cat_features_list=["gender", "age", "occupation", "zip_code"]),
    HistoryBasedFeaturesProcessor(
        use_log_features=True,
        use_conditional_popularity=True,
        user_cat_features_list=["gender", "age", "occupation", "zip_code"],
        item_cat_features_list=['title', 'release_date', 'imdb_url', 'unknown',
                                'Action', 'Adventure', 'Animation', "Children's",
                                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                                'Sci-Fi', 'Thriller', 'War', 'Western']
    )
])
def test_transformer_save_load(spark_sess: SparkSession, artifacts: ArtifactPaths, transformer, ctx):
    test_df = (
        artifacts.test
        .join(artifacts.user_features, on=["user_idx"], how="left")
        .join(artifacts.item_features, on=["item_idx"], how="left")
    )
    if isinstance(transformer, HistoryBasedFeaturesProcessor):
        transformer.fit(artifacts.train, artifacts.user_features, artifacts.item_features)
    else:
        transformer.fit(artifacts.train, artifacts.user_features)

    result = transformer.transform(test_df)
    assert result.count() == artifacts.test.count()

    path = os.path.join(artifacts.base_path, "some_transformer")

    save_transformer(transformer, path)
    loaded_transformer = load_transformer(path)

    new_result = loaded_transformer.transform(test_df)
    assert sorted(result.columns) == sorted(new_result.columns)


@pytest.mark.parametrize('ctx', ['test_data_splitting__out'], indirect=True)
@pytest.mark.parametrize('transformer', [
    ToNumericFeatureTransformer()
])
def test_feature_transformer_save_load(spark_sess: SparkSession, artifacts: ArtifactPaths, transformer, ctx):
    transformer.fit(artifacts.user_features)

    result = transformer.transform(artifacts.user_features)
    assert result.count() == artifacts.user_features.count()

    path = os.path.join(artifacts.base_path, "some_transformer")

    save_transformer(transformer, path)
    loaded_transformer = load_transformer(path)

    new_result = loaded_transformer.transform(artifacts.user_features)
    assert sorted(result.columns) == sorted(new_result.columns)


@pytest.mark.parametrize('ctx', ['test_data_splitting__out'], indirect=True)
def test_simple_dag_presplit_data(spark_sess: SparkSession, artifacts: ArtifactPaths, ctx):
    presplit_data.function(artifacts)

    assert os.path.exists(artifacts.first_level_train_path)
    assert os.path.exists(artifacts.second_level_positives_path)

    df = spark_sess.read.parquet(artifacts.first_level_train_path)
    assert 'user_idx' in df.columns and 'item_idx' in df.columns
    assert df.count() > 0

    df = spark_sess.read.parquet(artifacts.second_level_positives_path)
    assert 'user_idx' in df.columns and 'item_idx' in df.columns
    assert df.count() > 0


@pytest.mark.parametrize('ctx', ['test_simple_dag_presplit_data__out'], indirect=True)
def test_simple_dag_fit_predict_first_level_model(spark_sess: SparkSession, artifacts: ArtifactPaths, ctx):
    # # alternative 1
    model_class_name = "replay.models.knn.ItemKNN"
    model_kwargs = {"num_neighbours": 10}

    # alternative 2
    # model_class_name = "replay.models.als.ALSWrap"
    # model_kwargs = {"rank": 10, "seed": 42, "nmslib_hnsw_params": dense_hnsw_params}

    fit_predict_first_level_model(
        artifacts=artifacts,
        model_class_name=model_class_name,
        model_kwargs=model_kwargs,
        k=10
    )

    assert artifacts.partial_train_path(model_class_name)
    assert artifacts.model_path(model_class_name)
    assert artifacts.partial_predicts_path(model_class_name)

    # check train properties
    df = spark_sess.read.parquet(artifacts.partial_train_path(model_class_name))
    assert 'user_idx' in df.columns and 'item_idx' in df.columns and 'target' in df.columns
    assert len([c for c in df.columns if c.startswith('rel_')]) == 1
    assert df.count() > 0

    # check predict properties
    df = spark_sess.read.parquet(artifacts.partial_predicts_path(model_class_name))
    assert 'user_idx' in df.columns and 'item_idx' in df.columns and 'target' not in df.columns
    assert len([c for c in df.columns if c.startswith('rel_')]) == 1
    assert df.count() > 0

    # check the model
    # setattr(replay.model_handler, 'EmptyRecommender', EmptyRecommender)
    # setattr(replay.model_handler, 'PartialTwoStageScenario', PartialTwoStageScenario)
    # model = load(artifacts.partial_two_stage_scenario_path(model_class_name))
    # assert model is not None
    # assert type(model).__name__ == 'PartialTwoStageScenario'
    # model = cast(PartialTwoStageScenario, model)
    # assert len(model.first_level_models) == 1
    # fl_model = model.first_level_models[0]
    # full_type_name = f"{type(fl_model).__module__}.{type(fl_model).__name__}"
    # assert full_type_name == model_class_name
    #
    # if model_class_name.split('.')[-1] in ['ALSWrap', 'Word2VecRec'] and "nmslib_hnsw_params" in model_kwargs:
    #     assert os.path.exists(artifacts.hnsw_index_path(model_class_name))
