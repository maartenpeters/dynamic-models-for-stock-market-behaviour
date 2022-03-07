import re, yaml, os
import pandas as pd

def to_snake_case(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    name = re.sub('[^A-z0-9]+|\^', r'_', name)
    return name.lower()

def parse_data(data, grouping="policy_type_code", keys = ["notes", "policy_type_display", "policyvalue_actual"]):
        try:
            return [{_[grouping]: [{k: re.sub(r'((\r)|(\n)|(\\)|\s+)+', ' ', str(_[k]), flags=re.M)} for k in keys] + [{k: v} for k, v in data["stringencyData"].items()]} for _ in data["policyActions"] if _["notes"] is not None and _["notes"] != '']
        except:
            return None

def get_measure(df, date):
    small_date = str(date)[:10]
    parsed_data = parse_data(df[df["date"] == small_date].data.to_list()[0])
    return {small_date: parsed_data} if parsed_data != [] else None

def list_government_measures(dates):
    data_dir = "../data/covidtracker/"
    df = pd.DataFrame()
    for f in [f for f in os.listdir(data_dir) if f.split(".")[-1] == "parquet" and f.split(".")[0] in dates]:
        df = pd.concat([df, pd.read_parquet(f"{data_dir}/{f}")], axis=0)
    return yaml.dump([get_measure(df, date) for date in df.date.to_list() if get_measure(df, date) is not None])