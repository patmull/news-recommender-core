import optuna
import pandas as pd
import seaborn
from lightgbm import LGBMRanker
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.recommender_core.recommender_algorithms.hybrid import evaluation_results

SEED = 2021


def get_results_single_coeff_user_as_query():
    evaluation_results_df = evaluation_results.get_admin_evaluation_results_dataframe()
    print("evaluation_results_df:")
    print(evaluation_results_df)
    dict_of_jsons = {}
    for index, row in evaluation_results_df.iterrows():
        dict_of_jsons[row['user_id']] = row['results_part_2']

    print("dict_of_jsons:")
    print(dict_of_jsons)
    dataframes = []
    for id, json_dict in dict_of_jsons.items():
        df_from_json = pd.DataFrame.from_dict(json_dict)
        print("df_from_json:")
        print(df_from_json.to_string())
        df_from_json['user_id'] = id
        dataframes.append(df_from_json)
    df_merged = pd.concat(dataframes, ignore_index=True)

    print("df_merged columns")
    print(df_merged.columns)

    df_merged = df_merged[['user_id', 'slug', 'coefficient', 'relevance']]
    # converting indexes to columns
    # df_merged.reset_index(level=['coefficient', 'relevance'], inplace=True)
    print("df_merged:")
    print(df_merged.to_string())
    print("cols:")
    print(df_merged.columns)
    print("index:")
    print(df_merged.index)
    return df_merged


def recommend_posts():
    objective = 'lambdarank'
    features = ['coefficient', 'rating_count', 'rating_mean']

    df_results = get_results_single_coeff_user_as_query()

    train, test = train_test_split(df_results, test_size=0.2, random_state=SEED)
    print('train shape: ', train.shape)
    print('tests shape: ', test.shape)
    user_col = 'user_id'
    item_col = 'slug'
    target_col = 'relevance'
    train = train.sort_values('user_id').reset_index(drop=True)
    test = test.sort_values('user_id').reset_index(drop=True)
    # model query data
    train_query = train[user_col].value_counts().sort_index()
    test_query = test[user_col].value_counts().sort_index()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED)  # fix random_order seed
                                )
    study.optimize(objective, n_trials=10)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    best_params = study.best_trial.params
    model = LGBMRanker(n_estimators=1000, **best_params, random_state=SEED)
    model.fit(
        train[features],
        train[target_col],
        group=train_query,
        eval_set=[(test[features], test[target_col])],
        eval_group=[list(test_query)],
        eval_at=[1, 3, 5, 10, 20],
        early_stopping_rounds=50,
        verbose=10
    )

    TOP_N = 20
    model.predict(test.iloc[:TOP_N][features])

    # feature imporance
    plt.figure(figsize=(10, 7))
    df_plt = pd.DataFrame({'feature_name': features, 'feature_importance': model.feature_importances_})
    df_plt.sort_values('feature_importance', ascending=False, inplace=True)
    seaborn.barplot(x="feature_importance", y="feature_name", data=df_plt)
    plt.title('feature importance')

    return model
