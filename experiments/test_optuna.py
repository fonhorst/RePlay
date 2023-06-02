import optuna


def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2


study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=22))
study.optimize(objective, n_trials=5)

print(study.best_params)  # E.g. {'x': 2.002108042}

# study2 = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=22))
# study2.optimize(objective, n_trials=10)

study2 = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=22))
study2.optimize(objective, n_trials=2)
print(study2.best_params)
study2.optimize(objective, n_trials=3)
print(study2.best_params)  # E.g. {'x': 2.002108042}

# import pickle
#
# # Save the sampler with pickle to be loaded later.
# with open("sampler.pkl", "wb") as fout:
#     pickle.dump(study.sampler, fout)
#
# restored_sampler = pickle.load(open("sampler.pkl", "rb"))
# study = optuna.create_study(
#     study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
# )
# study.optimize(objective, n_trials=3)

if study.best_params == study2.best_params:
    print("identical results")
else:
    print("Not identical results")

# {'x': 1.5021450274563009}
# {'x': 2.878374668112871}

