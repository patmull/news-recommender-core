import json


def flatten(multi_dimensional_list):
    return [item for sublist in multi_dimensional_list for item in sublist]


def convert_to_json(df):
    predictions_json = df.to_json(orient="split")
    predictions_json_parsed = json.loads(predictions_json)
    return predictions_json_parsed
