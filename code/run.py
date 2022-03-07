from datetime import datetime, timedelta
from utils import *
# Models
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from pmdarima import arima
import pmdarima
# Feature selection
from sklearn.feature_selection import RFECV
# Pipeline
from sklearn.pipeline import Pipeline
from pmdarima import pipeline
# Preprocessing
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
from pmdarima import preprocessing as ppc
# Optimization
from sklearn.model_selection import GridSearchCV
from pmdarima import model_selection
# Scoring
from sklearn import metrics
import pandas as pd, numpy as np, pmdarima as pm
import sys, argparse, logging

__version__ = '0.1.0'
__author__ = u'Maarten Peters'

ts = str(str(datetime.now())[:16]).replace(':','_').replace('-','_')

logging.basicConfig(filename=f'logs/model_run_{ts}.log', encoding='utf-8', level=logging.DEBUG)
logging.info(f'Running models started at {ts}')

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser('ModelRunner')
    version = '%(prog)s ' + __version__
    parser.add_argument('--version', '-v', action='version', version=version)
    parser.add_argument('--n_samples', '-n', type=int, help='Number of samples to be taken', default=5, )
    parser.add_argument('--period', '-p', type=str, help='Period type of the data either d for dialy or w for weekly', default='d', )
    parser.add_argument('--length_prediction', '-l', type=int, help='Number of prediction points to generate', default=7, )
    parser.add_argument('--seed', '-s', type=int, help='Integer value for seed', default=0, )
    parser.add_argument('--response_variable', '-r', type=str, help='Response variable to be predicted, either "open", "close", "high" or "low" or comma separated list of either', default="open")
    parser.add_argument('--train_test_ratio', '-t', type=float, help='Ratio of training data vs. testing data', default=float(10/1))
    parser.add_argument('--models', '-m', type=str, help='Comma separated string of model abbreviations to run', default="lr,rfr,sgd,dummy,arima")
    parser.add_argument('--input_blacklist', '-i', type=str, help='Comma separated string of blacklisted input columns', default="")
    parser.add_argument('--all_stocks', '-a', type=str, help='Sample all stocks: "true" or "false"', default="false")
    return parser

# Sample single stock
def sample_single_stock(df, random_state, sample_stock=None, retries=3):
    # try:
        # if retries <0:
        #     raise "Broken!"
    if sample_stock is None:
        sample_stock = df.sample(random_state=random_state).index.get_level_values("ticker").to_list()[0]
    idx = pd.IndexSlice
    df = df.loc[idx[:, [sample_stock]], :]
    # except:
    #     df = sample_single_stock(df, random_state, sample_stock, retries=retries-1)
    return df.reset_index(level=[0,1]).set_index("date"), sample_stock

def sample_time_series(df, random_state, length_prediction, train_test_ratio, period, retries=3):
    try:
        if retries < 0:
            raise "Broken!"
        train_delta = int((length_prediction - 1) * train_test_ratio * period)
        train_week_start = str(df[(df.index + timedelta(int(train_delta * 1.2))) < max(df.index)].sample(random_state=random_state)["dt_iso_year_week"].to_list()[0])
        train_day_start = df[(df["dt_iso_year_week"] == train_week_start) & (df["dt_day_of_week"] == 0)].index[0]
        if (max(df.index) - train_day_start).days < int(train_delta * 1.2):
            train_start, train_end, test_start, test_end, val_start, val_end = sample_time_series(df, random_state, length_prediction, train_test_ratio, period, retries=retries-1)
        else:
            test_end = int(train_delta * (1 + 1/train_test_ratio))
            intervals = [0, train_delta - 1, train_delta, test_end, test_end + 1, test_end + (test_end - train_delta)]
            train_start, train_end, test_start, test_end, val_start, val_end = [train_day_start + timedelta(t) for t in intervals]
    except:
        train_start, train_end, test_start, test_end, val_start, val_end = sample_time_series(df, random_state, length_prediction, train_test_ratio, period, retries=retries-1)
    return train_start, train_end, test_start, test_end, val_start, val_end

def train_test_val_split(X, y, train_start, train_end, test_start, test_end, val_start=None, val_end=None):
    X_train = X.loc[train_start:train_end].reset_index(drop=True)
    y_train = y.loc[train_start:train_end].reset_index(drop=True)
    X_test = X.loc[test_start:test_end].reset_index(drop=True)
    y_test = y.loc[test_start:test_end].reset_index(drop=True)
    if val_start is None or val_end is None:
        X_val = X.loc[val_start:val_end].reset_index(drop=True)
        y_val = y.loc[val_start:val_end].reset_index(drop=True)
    else:
        X_val, y_val = [None] * 2
    return X_train, y_train, X_test, y_test, X_val, y_val

def load_data(path = "../data/preprocessed/combined.parquet"):
    return pd.read_parquet(path)

def add_frac(x):
    return np.add(x, 1e-6)

def clean(x):
    x = x.fillna(0)
    x[x == -np.inf] = 0
    x[x == np.inf] = 0
    return x

def aggregate_data(df):
    df = df[df["is_original_stockdata"] == True]
    df = df.drop([col for col in df.columns if "dt_" in col], axis=1)
    df['date'] = pd.to_datetime(df.index) - pd.to_timedelta(7, unit='d')
    agg_cols = [col for col in df.columns if col != "date"]
    return df.groupby(["ticker", pd.Grouper(key='date', freq='W-MON')])[agg_cols].mean().reset_index().sort_values('date').set_index("date")

def build_and_fit(model_name, model, X_train, y_train, period, parameters={}, *args, **kwargs):
    n_cols = X_train.shape[0] + y_train.shape[0]
    col_slice = slice(0,n_cols)
    if parameters is None:
        paramaters = {}
    if model_name == "arima":
        m = 52 if period == 7 else 7
        pipe = pmdarima.pipeline.Pipeline([
            ("ppc", ppc.FourierFeaturizer(m=m)),
            (model_name, model(*args, **kwargs))
        ])
        fit = pipe.fit(X=X_train, y=y_train)
        best_parameters = None
    elif model_name == "dummy":
        pipe = Pipeline([
            (model_name, model(*args, **kwargs))
        ])
        fit = pipe.fit(X_train, y_train)
        best_parameters = None 
    else:
        pipe = Pipeline([
            ('feature_selection', RFECV(model(*args, **kwargs))), #, importance_getter="auto")),
            (model_name, model(*args, **kwargs))
        ])
        grid_search = GridSearchCV(pipe, param_grid=parameters, n_jobs=-1, verbose=1)
        fit = grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_estimator_.get_params()   
    return fit, best_parameters

def get_expl_vars(df_cols, response_vars, blacklist = []):
    blacklist += ["ticker", "volume", 'is_original_stockdata', 'is_original_jhu']
    blacklist += [f"{y_col}_{incdec}" for y_col in response_vars + ["volume"] for incdec in ["is_equal", "is_increase", "is_decrease"]]
    blacklist += [col for col in df_cols if "dt_" in col]
    blacklist += [col for col in df_cols if "policyvalue_actual" in col]
    expl_vars = [col for col in df_cols if col not in response_vars and col not in blacklist]
    return sorted(list(set(expl_vars)))

def load_models(random_state):
    models = {
        "lr": {
            "model": LinearRegression,
            "kwargs": None,
            # "params": {
            #     "normalize": [True, False]
            # }e
            "params": {}
        },
        "sgd": {
            "model": SGDRegressor,
            "kwargs": {
                "max_iter": 10000,
                "random_state": random_state,
            },
            "params": {}
        },
        "rfr": {
            "model": RandomForestRegressor,
            "kwargs": {
                "random_state": random_state,
            },
            "params": {
                "n_estimators": [10, 20, 100]
            }
        },
        "arima": {
            "model": arima.AutoARIMA,
            "kwargs": {
                "random": True,
                "random_state": random_state,
                "stepwise": True, 
                "trace": 1, 
                "error_action": "ignore",
                "seasonal": False,  # because we use Fourier
                "suppress_warnings": True,
                "n_fits": 20,
                "max_p": 12, 
                "max_q": 12, 
                "max_P": 12, 
                "max_Q": 12,
            },
            "params": {}
        },
        "dummy": {
            "model": DummyRegressor,
            "kwargs": {
                "strategy": "constant",
            },
            "params": {}
        }
    }
    for model_name, model_value in models.items():
        model_value["features"] = {
            "X": None,
            "Y": None
        }
        models[model_name] = model_value
    return models

def log_transform(x):
    return np.log(x + 1)

def main(args=None):
    # Get cmd line arguments
    parser = get_parser()
    args = parser.parse_args(args)
    
    # Random state
    seed = args.seed
    random_state = np.random.RandomState(seed)
    
    n_samples = args.n_samples
    response_variables = [_ for _ in args.response_variable.split(",")]
    all_response_variables = ["open", "high", "low", "close"]
    period = 7 if args.period == "w" else 1
    length_prediction = args.length_prediction
    train_test_ratio = args.train_test_ratio
    input_blacklist = args.input_blacklist.split(",")
    all_stocks = True if args.all_stocks == "true" else False
    models = load_models(random_state)
    logging.info(f"{str(datetime.now())}: Loaded models: {', '.join(models.keys())}")
    
    # Load main data
    main_df = load_data()
    logging.info(f"{str(datetime.now())}: Running main function, init with random state {str(seed)} for {n_samples} n_samples.")
    results = []
    
    for model_key, model_val in models.items():
        if model_key not in args.models.split(","):
            logging.info(f"{str(datetime.now())}: Skipping model {model_key}")
            continue
        for i in range(n_samples):
            df = main_df.copy()
            logging.info(f"{str(datetime.now())}: Starting run {i}, model {model_key} for response variables '{', '.join(response_variables)}' with dataframe size {df.shape}.")

            # Sample ticker
            if all_stocks:
                sample_stocks = list(set([s for s in main_df.index.get_level_values("ticker").to_list() if type(s) == str]))
            else:
                sample_stocks = [None]
            for sample_stock in sample_stocks:
                # Log current run
                current_run = {
                    "id": i,
                    "model": model_key,
                    "response_vars": response_variables
                }
                df, sample_stock = sample_single_stock(main_df, random_state, sample_stock)
                non_zero = df[df[response_variables[0]] != 0]
                non_zero_start, non_zero_end = non_zero[response_variables[0]].index.min(), non_zero[response_variables[0]].index.max()
                df = df.loc[non_zero_start:non_zero_end]
                current_run["ticker"] = sample_stock
                logging.info(f"{str(datetime.now())}: Sampled stock '{sample_stock}'.")

                # Sample time period
                train_start, train_end, test_start, test_end, val_start, val_end = sample_time_series(df, random_state, length_prediction, train_test_ratio, period)
                logging.info(f"{str(datetime.now())}: Sampled start and end dates; train start: {train_start}, train end: {train_end}, test start: {test_start}, test end: {test_end}")

                current_run["train_start"] = train_start
                current_run["train_end"] = train_end
                current_run["test_start"] = test_start
                current_run["test_end"] = test_end
                current_run["val_start"] = val_start
                current_run["val_end"] = val_end

                # Aggregate data
                if period == 7:
                    try:
                        df = aggregate_data(df)
                        logging.info(f"{str(datetime.now())}: Aggregated data to size '{df.shape}'.")
                    except:
                        logging.info(f"{str(datetime.now())}: Skipping data aggregation for run {i}, model {model_key} and ticker {sample_stock}.")
                        continue

                # Split X and y
                y_cols = current_run["response_vars"]
                blacklist = input_blacklist
                X_cols = get_expl_vars(df.columns, all_response_variables, blacklist = blacklist)
                X, y = df[X_cols].copy(), df[y_cols].copy()
                for col in X.columns:
                    X[f"{col}_diff"] = X[col].diff().fillna(0)
                current_run["expl_vars"] = X.columns.to_list()
                logging.info(f"{str(datetime.now())}: Set explanatory variables to '{', '.join(X_cols)}'.")

                # Set date to ordinal for correct predictions
                X["date"] = X.index.get_level_values("date")
                X["date"] = X["date"].map(datetime.toordinal)
                
                # Add dummy column for simple regression
                X["dummy"] = 1.0

                # Preprocess data for training and testing
                transform_pipeline =  Pipeline([
                    ('minmax', MinMaxScaler((1,1000))),
                    ('log', FunctionTransformer(log_transform)),
                    ('minmax2', MinMaxScaler((1,1000))),
                    ('normality_transform', PowerTransformer()),   
                    ('scaler', MinMaxScaler()),
                ])
                X_index, X_columns = X.index, X.columns
                X = pd.DataFrame(transform_pipeline.fit_transform(X, y), columns=X_columns, index=X_index)

                # Split X and y in train and test
                X_train, y_train, X_test, y_test, _, _ = train_test_val_split(X, y, train_start, train_end, test_start, test_end, val_start, val_end)
                logging.info(f"{str(datetime.now())}: Training and testing size set to: 'X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}, '.")

                # Build pipeline and fit model
                if model_val["features"]["X"] is None:
                    model_val["features"]["X"] = X_cols
                if model_val["features"]["Y"] is None:
                    model_val["features"]["Y"] = y_cols
                if model_val["params"] == {}:
                    params = {}
                    model_val["params"] = params
                if model_val["kwargs"] is not None:
                    model_kwargs = model_val["kwargs"]
                else:
                    model_kwargs = {}
                model = model_val["model"]

                # PCT removed from baseline models for functionality issues
                y_train = y_train.values.ravel()
                y_test = y_test.values.ravel()
                if model_key == "dummy":
                        model_kwargs["constant"] = y_train[-1]
                try:
                    pipe, params = build_and_fit(current_run["model"], model, X_train, y_train, period, parameters={}, **model_kwargs)
                    current_run["is_valid"] = True
                except:
                    logging.info(f"{str(datetime.now())}: Ran into critival error fitting model '{current_run['model']}'")
                    current_run["is_valid"] = False
                current_run["params"], model_val["params"] = [params] * 2
                logging.info(f"{str(datetime.now())}: Fitted model '{current_run['model']}'")

                # Build prediction
                current_run["y_test"] = y_test
                if current_run["is_valid"] == True:
                    if current_run["model"] != "arima":
                        y_pred = pipe.predict(X=X_test)
                    else:
                        y_pred = pipe.predict(n_periods=length_prediction, X=X_test)
                    current_run["y_pred"] = y_pred
                    logging.info(f"{str(datetime.now())}: Predictions made with size: 'y_pred: {y_pred.shape}'")

                # Score model
                try:
                    current_run["train_results"] = pipe.cv_results_
                except:
                    current_run["train_results"] = None
                try:
                    current_run["test_score"] = metrics.mean_absolute_percentage_error(y_test, y_pred) # pipe.score(X_test, y_test)
                except:
                    current_run["test_score"] = None
                logging.info(f"{str(datetime.now())}: Model scored: {str(current_run['test_score'])}'")
                try:
                    current_run["is_valid"] = True if pipe is not None or current_run["is_valid"] else False
                except:
                    current_run["is_valid"] = False

                # Add current run to results
                results.append([current_run])
        
    # Dump results to json file
    with open(f"./results/model_run_{seed}_{args.period}_{'_'.join(sorted(models.keys()))}_{args.response_variable.replace(',','_')}_{ts}.json", "w+") as results_file:
        pd.DataFrame(results).to_json(results_file)

if __name__ == "__main__":
    main()