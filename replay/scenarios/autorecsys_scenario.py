from replay.scenarios import TwoStagesScenario
from replay.splitters import Splitter, UserSplitter


class AutoRecSysScenario:

    """

    AutoRecSys scenario which construct training pipeline and return the best model combination: 1stage either two-stage

    """

    def __init__(self, task, subtask, data, time):

        self.task = task
        self.subtask = subtask
        self.data = data
        self.time = time

    def get_scenario(self):

        MODEL_NAMES2MODELS = {}
        first_level_models_names_default = ["ALS", "SLIM", "Item_KNN", "Word2Vec"]
        first_level_models_names_sparse = ["Item_KNN", "ALS",  "Word2Vec", "SLIM"]

        # TODO: add heuristics for choosing the most appropriate scenario
        # 1 one-stage: default hyper parameters 100% of train
        scenario = TwoStagesScenario(
            )

        # 2 one-stage: optimized hp 100% of train
        scenario = TwoStagesScenario()

        # 3 two-stage: default hyper parameters 80-20 for 1st-2nd level

        scenario = TwoStagesScenario(train_splitter=UserSplitter(
            item_test_size=0.2, shuffle=True, seed=42),
            first_level_models=[MODEL_NAMES2MODELS[model_name] for model_name in first_level_models_names_default],
            custom_features_processor=None)

        # 4 two-stage: optimized hp 80-20 for 1st-2nd level

        do_optimization = True

        return scenario, do_optimization

    def fit(self):

        scenario, do_optimization = self.get_scenario()

        if do_optimization:
            best_params = scenario.optimize()
            scenario.first_level_models = [m for m in scenario.first_level_models]  # TODO: add best params here

        scenario.fit()

    def predict(self):
        pass

    def fit_predict(self):
        pass


        # TODO [opt]: re-split heuristics
