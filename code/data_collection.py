import requests, re, os, shutil
from bs4 import BeautifulSoup
from datetime import datetime
from git import Repo
import pandas as pd
import yfinance as yf
from utils import to_snake_case

class DataCollector():
    """
    A class that contains all data collection methods.
    Parameters
    --------------
    root_dir : str
        Root directory for the data
    log_dir : str
        Directory for logging
    """
    def __init__(
        self, 
        root_dir: str = "./data", 
        log_dir: str = "./logs", 
        datasets: list = {"rivm": None, "stocks": None, "jhu": None, "weather": None,  "measures": None}, 
        start_date: str = "2020-01-01",
        end_date: str = str(datetime.now())[:10]
    ):
        self.root_dir = root_dir
        self.log_dir = log_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.datasets = datasets
        self.start_date = start_date
        self.end_date = end_date
        
    def get(self, dataset: str = None, *args, **kwargs):
        """
        Collects data from dataset, passing arguments to hidden methods
        Parameters
        --------------
        dataset : str
            Dataset name, either:
            - "rivm"
            - "stocks"
            - "jhu"
            - "weather"
            - "measures"
            - "all"
        *args
        **kwargs
        
        Returns
        --------------
        DataFrame
            Returns Pandas DataFrame or list of DataFrames with data
        """
        if dataset == "rivm":
            return self._get_rivm(*args, **kwargs)
        elif dataset == "stocks":
            return self._get_stocks(*args, **kwargs)
        elif dataset == "jhu":
            return self._get_jhu(*args, **kwargs)
        elif dataset == "weather":
            return self._get_weather(*args, **kwargs)
        elif dataset == "measures":
            return self._get_measures(*args, **kwargs)
        elif dataset == "all":
            return {ds: self.get(ds, *args, **kwargs) for ds in self.datasets.keys()}
            
        else:
            return None
        
    def write_to_log(self, msg):
        """
        Writes data to logfile "logs.txt"
        """
        with open(os.path.join(self.log_dir, "logs.txt"), "a+") as log_file:
            log_file.write(f"{str(datetime.now())}: {msg}\n")
            return msg
    
    def file_to_parquet(self, df: pd.DataFrame = None, file_name: str = "data.csv"):
        """
        Writes Pandas DataFrame to parquet
        """
        new_file = os.path.splitext(file_name)[0] + ".parquet"
        print(f"Compressing {file_name} to parquet")
        df.to_parquet(new_file)
    
    def file_from_parquet(self, file_name: str = "data.csv"):
        """
        Reads data from parquet
        """
        new_file = os.path.splitext(file_name)[0] + ".parquet"
        print(f"Collection {file_name} from parquet")
        return pd.read_parquet(new_file)
        
    def _log_get(self, file_name, from_loc, to_loc):
        """
        Prints file collection logs
        """
        print(self.write_to_log(f"Collecting file {file_name} from {from_loc} into {to_loc}"))
    
    def _get_rivm(self, subdir: str = "rivm", subset: str = "aantallen_gemeente_per_dag"):
        """
        Collects data from the RIVM, with an optional subset
        Parameters
        --------------
        subdir : str
            Subdirectory for data storage, default "rivm"
        subset : str
            Optional subset selection for data loading. Accepts:
            - "aantallen_gemeente_cumulatief"
            - "aantallen_gemeente_per_dag"
            - "aantallen_settings_per_dag"
            - "casus_landelijk"
            - "gedrag"
            - "gehandicaptenzorg"
            - "ic_opnames"
            - "rioolwaterdata"
            - "thuiswonend_70plus"
            - "uitgevoerde_testen"
            - "verpleeghuizen"
            - "ziekenhuis_ic_opnames_per_leeftijdsgroep"
            - "ziekenhuisopnames"

        Returns
        --------------
        DataFrame
            Returns Pandas DataFrame with data
        """
        # Base parameters
        base_url = 'https://data.rivm.nl/covid-19/'
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        pat = re.compile(r'COVID\-19.+\.csv')
        csv_anchors = soup.find_all('a', href=pat)
        csv_urls = [{'file_name':_.get('href'), 'url':f"{base_url}{_.get('href')}"} for _ in csv_anchors if subset in _.get('href')]
        # Store files
        for csv_url in csv_urls:
            sub_path = os.path.join(self.root_dir, subdir)
            dest_file = os.path.join(sub_path, csv_url['file_name'])
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
            self._log_get(csv_url["file_name"], base_url, dest_file)
            with open(dest_file, 'wb') as file_obj:
                file_obj.write(requests.get(csv_url['url']).content)
            df = pd.read_csv(dest_file, sep=";")
            self.file_to_parquet(df, dest_file)
        return self.file_from_parquet(dest_file)
    
    def _get_stocks(
        self, 
        subdir: str = "yfinance", 
        tickers = ["^AEX","RDSA.AS","CL=F"], 
        period: str = "max", 
        interval: str = "1d", 
        group_by: str = 'ticker', 
        auto_adjust: bool = True, 
        prepost: bool = True, 
        threads: bool = True, 
        proxy: bool = None, 
        *args, 
        **kwargs
    ):
        """
        Collects data from the Yahoo Finance API
        Parameters
        --------------
        subdir : str
            Subdirectory for data storage, default "yfinance"
        tickers : list or str
            tickers list or string as well
        period : str
            use "period" instead of start/end
            valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            (optional, default is '1mo')
        interval: str
            fetch data by interval (including intraday if period < 60 days)
            valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            (optional, default is '1d')
        group_by: str
            group by ticker (to access via data['SPY'])
            (optional, default is 'column')
        auto_adjust: bool
            adjust all OHLC automatically
            (optional, default is False)
        prepost: bool
            download pre/post regular market hours data
            (optional, default is False)
        threads: bool
            use threads for mass downloading? (True/False/Integer)
            (optional, default is True)
        proxy: str
            proxy URL scheme use use when downloading?
            (optional, default is None)
        Returns
        --------------
        DataFrame
            Returns Pandas DataFrame with data
        """
        df = yf.download(
            tickers = tickers,
            period = period,
            interval = interval,
            group_by = group_by,
            auto_adjust = auto_adjust,
            prepost = prepost,
            threads = threads,
            proxy = proxy,
             *args, **kwargs
        )
        full_df = pd.DataFrame()
        for t in list(set([_[0] for _ in df.columns])):
            ticker_df = df[t].copy()
            ticker_df["Ticker"] = t
            full_df = pd.concat([full_df, ticker_df], axis=0)
        sub_path = os.path.join(self.root_dir, subdir)
        dest_file = os.path.join(sub_path, "data")
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        self._log_get(f"{subdir}", "yfinance", dest_file)
        self.file_to_parquet(full_df, dest_file)
        return full_df
    
    def _get_weather(
        self, 
        subdir: str = "weather",
        url: str = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens", 
        options = "stns=260&vars=ALL&byear=2019&bmonth=1&bday=1&eyear=2021&emonth=12&eday=31&fmt=json"
    ):
        """
        Collects weather data from the KNMI
        Parameters
        --------------
        git_repo : str
            The git repository to collect
        subdir : str
            Subdirectory for data storage, default "weather"
        url : str
            Optional URL for data colllection
        options : str
            Parameters for 
        Returns
        --------------
        DataFrame
            Returns Pandas DataFrame with data
        """
        sub_path = os.path.join(self.root_dir, subdir)
        dest_file = os.path.join(sub_path, "knmi.json")
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        self._log_get(subdir, url, dest_file)
        with open(dest_file, 'wb') as file_obj:
            file_obj.write(requests.get(url, params=options).content)
        df = pd.read_json(dest_file)
        self.file_to_parquet(df, dest_file)
        return self.file_from_parquet(dest_file)
    
    def _get_measures(
        self,
        subdir: str = "covidtracker",
        url: str = "https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/actions",
        region: str = "NLD"
    ):
        base_url = f"{url}/{region}"
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        dfs = []
        for date in date_range:
            date_str = str(date)[:10]
            try:
                df = pd.read_parquet(f"./{self.root_dir}/{subdir}/{date_str}.parquet")
            except:
                response = requests.get(f"{base_url}/{date_str}/")
                df = pd.DataFrame(
                    {
                        "date":[date],
                        "data":[response.json()]
                    }
                )
                df.to_parquet(f"./{self.root_dir}/{subdir}/{date_str}.parquet")
                self._log_get(subdir, base_url, f"{date_str}.parquet")
            finally:
                dfs += [df]
        return pd.concat(dfs)
        
    def _get_jhu(self, git_repo: str = "https://github.com/govex/COVID-19.git", subdir: str = "govex", subset: str = "time_series_covid19_vaccine_global"):
        """
        Collects data from the Johns Hopkins University COVID-19 Github page
        Parameters
        --------------
        git_repo : str
            The git repository to collect
        subdir : str
            Subdirectory for data storage, default "govex"
        subset : str
            Optional subset selection for data loading
        Returns
        --------------
        DataFrame
            Returns Pandas DataFrame with data
        """
        # !Note: Deprecated in favor of Git submodule
        sub_path = os.path.join(self.root_dir, subdir)
        # if os.path.exists(sub_path):
        #     shutil.rmtree(sub_path)
        # Repo.clone_from(url=git_repo, to_path=sub_path)
        dest_file = os.path.join(sub_path, "data_tables", "vaccine_data", "global_data", f"{subset}.csv")
        self._log_get(f"{dest_file}.csv", git_repo, dest_file)
        df = pd.read_csv(dest_file)
        return df
    

class Preprocessor():
    """
    A class that preprocesses all data
    Parameters
    --------------
    root_dir : str
        Root directory for the data
    log_dir : str
        Directory for logging
    datasets : dict
        Dict of dicts with dataset names and Pandas DataFrames, default: []
    """
    def __init__(
        self, 
        root_dir: str = "./data", 
        log_dir: str = "./logs", 
        datasets: dict = {},
        start_date: str = "2020-01-01",
        end_date: str = str(datetime.now())[:10]
    ):
        self.root_dir = root_dir
        self.log_dir = log_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.datasets = datasets
        self.start_date = start_date
        self.end_date = end_date
    
    def preprocess(self, df: pd.DataFrame, dataset: str = None, *args, **kwargs):
        """
        Collects data from dataset, passing arguments to hidden methods
        Parameters
        --------------
        dataset : str
            Dataset name, either:
            - "rivm"
            - "stocks"
            - "jhu"
            - "weather"
            - "measures"
            - "dates"
        *args
        **kwargs
        
        Returns
        --------------
        DataFrame
            Returns Pandas DataFrame with data
        """
        if dataset == "rivm":
            return self._preprocess_rivm(df, *args, **kwargs)
        elif dataset == "stocks":
            return self._preprocess_stocks(df, *args, **kwargs)
        elif dataset == "jhu":
            return self._preprocess_jhu(df, *args, **kwargs)
        elif dataset == "weather":
            return self._preprocess_weather(df, *args, **kwargs)
        elif dataset == "measures":
            return self._preprocess_measures(df, *args, **kwargs)
        elif dataset == "dates":
            return self._preprocess_dates(*args, **kwargs)
        else:
            return None
    
    def combine(self, dfs: list = [], *args, **kwargs):
        df_dates = self.preprocess(None, "dates")
        for df in dfs.values():
            df_dates = df_dates.merge(df, how="outer", on="date")
        return df_dates
    
    def preprocess_and_combine(self, dfs: dict = {}, *args, **kwargs):
        if dfs == {} and self.datasets is not None:
            dfs = self.datasets
        else:
            dfs = {ds: DataCollector.get(dataset=ds) if ds != "stocks" else DataCollector.get(dataset=ds, tickers=TICKERS) for ds, v in dfs.items()}
        dfs = {ds: self.preprocess(df, ds) if ds != "stocks" else self.preprocess(df, ds, idxs=df["Ticker"].unique().tolist()) for ds, df in dfs.items()}
        df_dates = self.preprocess(None, "dates")
        dtypes = {k: str(v) for k, v in df_dates.dtypes.to_dict().items()}
        if dfs != {}:
            for ds, df in dfs.items():
                dtypes = {**dtypes, **{k: str(v) for k, v in df.dtypes.to_dict().items()}}
                df_dates = df_dates.merge(df.sort_index(), how="outer", on="date")
        for col, dtype in dtypes.items():
            if dtype not in ("object", "bool"):
                df_dates[col] = df_dates[col].fillna(0)              
            df_dates[col] = df_dates[col].astype(dtype)
        df_dates["date"] = df_dates.index
        # df_dates = df_dates[~df_dates["ticker"].isna()]
        df_dates = df_dates.set_index(["date", "ticker"])
        df_dates.to_parquet(f"{self.root_dir}/preprocessed/combined.parquet")
        return df_dates

    def _preprocess_rivm(self, df):
        # Subset columns
        df = df[['Date_of_publication', 'Total_reported', 'Deceased']]
        df.columns = ['date', 'infected', 'deceased']
        # Subset timerange
        df = df[df['date'] >= self.start_date]
        df = df[df['date'] <= self.end_date]
        df = df.groupby(by=['date']).sum()
        df['infected'], df['deceased'], df.index = df['infected'].astype('int64'), df['deceased'].astype('int64'), df.index.astype('datetime64[ns]')
        return df

    def _preprocess_stocks(self, df, idxs = ['GLPG.AS',
                                            'ASM.AS',
                                            'RDSA.AS',
                                            'KPN.AS',
                                            'REN.AS',
                                            'IMCD.AS',
                                            'RAND.AS',
                                            'UNA.AS',
                                            'AKZA.AS',
                                            'PHIA.AS',
                                            'AD.AS',
                                            'ABN.AS',
                                            'URW.AS',
                                            'TKWY.AS',
                                            'WKL.AS',
                                            'DSM.AS',
                                            'ASML.AS',
                                            'ADYEN.AS',
                                            'ASRNL.AS',
                                            'HEIA.AS',
                                            'INGA.AS',
                                            'MT.AS',
                                            'PRX.AS',
                                            'AGN.AS',
                                            'NN.AS']):
        stock_columns = ["open", "close", "low", "high", "volume"]
        full_df = pd.DataFrame()
        # Subset timerange
        df = df.loc[self.start_date:self.end_date, :]
        df.columns = [_.lower() for _ in df.columns]
        df.index.names = ["date"]
        for t in list(set(df.ticker.to_list())):
            # print(f"Processing ticker data for '{t}'")
            ticker_df = df[df["ticker"] == t]
            dates = df.index.tolist()
            # Forward fill dates
            ticker_df.resample("1D")
            ticker_df = ticker_df.fillna(method="ffill", axis=0)
            ticker_df = ticker_df.asfreq('1D', method="ffill")
            ticker_df["is_original_stockdata"] = ticker_df.index.isin(dates)
            for c in stock_columns:
                ticker_df[f"{c}_is_increase"] = ticker_df[f"{c}"] > ticker_df[f"{c}"].shift()
                ticker_df[f"{c}_is_decrease"] = ticker_df[f"{c}"] < ticker_df[f"{c}"].shift()
                ticker_df[f"{c}_is_equal"] = ticker_df[f"{c}"] == ticker_df[f"{c}"].shift()
            full_df = pd.concat([full_df, ticker_df], axis=0)
        full_df.index = full_df.index.astype('datetime64[ns]')
        # full_df["date"] = full_df.index
        # full_df = full_df.set_index(["date", "ticker"])
        return full_df
        
    def _preprocess_jhu(self, df, region: str = "Netherlands"):
        # Subset data and format
        df = df[(df["Country_Region"] == region) & (df['UID'] == 528)] # They added the ABC islands to the Netherlands, doh
        df = df.drop(labels=["UID", "Report_Date_String", "Country_Region", "Province_State"], axis=1)
        df = df.fillna(0)
        df = df.rename(columns={
            "Date": "date",
            "Doses_admin": "total_doses",
            "People_partially_vaccinated": "partial_vaccinations",
            "People_fully_vaccinated": "full_vaccinations"
        })
        df = df.astype({"date": "datetime64",
                        "total_doses": "int64",
                        "partial_vaccinations": "int64",
                        "full_vaccinations": "int64"})
        df = df.set_index("date")
        dates = df.index.tolist()
        # Subset timerange
        df = df.loc[self.start_date:self.end_date]
        df.index = df.index.astype('datetime64[ns]')
        # Forward fill dates
        df.resample("1D")
        df = df.fillna(method="ffill", axis=0)
        df = df.sort_index()
        df = df.asfreq('1D', method="ffill")
        df["is_original_jhu"] = df.index.isin(dates)
        df["daily_doses"] = df["total_doses"].diff()
        # df.set_index("date", inplace=True)
        return df
    
    def _preprocess_measures(self, df):
        # Data structure:
        #
        # {
        #   policyActions: {
        #     0...n: {  //Numerical key
        #       policy_type_code: String, //Policy type 2 or 3 digit code - letter/number - or NONE if no data available
        #       policy_type_display: String, //String describing policy value,
        #       policyvalue: Integer, //Represents policy status
        #       is_general: Boolean, //If this is a general policy,
        #       flagged: Boolean, //Replaces isgneral from 28 April 2020,
        #       policy_value_display_field: String, //Describes the level of stringency of the policy or type of policy
        #       notes: String, //Notes entered by contributors
        #     }
        #   },
        #   stringencyData: {
        #     date_value: String, //YYYY-MM-DD date of record
        #     country_code: String, //ALPHA-3 country code
        #     confirmed: Integer, //Recorded confirmed cases,
        #     deaths: Integer, //Recorded deaths,
        #     stringency_actual: Integer, //Calculated stringency
        #     stringency: Integer, //Display stringency - see notes **1 above
        #   }
        # }
        df = df.set_index("date")
        df = pd.concat([df.drop(['data'], axis=1), df['data'].apply(pd.Series)], axis=1)
        df = pd.concat([df.drop(['stringencyData'], axis=1), df['stringencyData'].apply(pd.Series)], axis=1)
        df = df.explode("policyActions")
        df = pd.concat([df.drop(['policyActions'], axis=1), df['policyActions'].apply(pd.Series)], axis=1)
        df = df[(~df["notes"].isna()) & (df["policyvalue_actual"] > 0)]
        df["date"] = df.index
        df["date"] = df["date"].astype("str").apply(lambda x: str(x)[:10])
        df = df.pivot(index=["date"], columns=["policy_type_code"], values=["policyvalue_actual", "stringency_actual"]).fillna(0)
        df.columns = [f"{_[1].lower()}_{to_snake_case(_[0].replace('.','_'))}" for _ in df.columns]
        df.index = df.index.astype('datetime64[ns]')
        stringency_cols = [col for col in df.columns if "stringency" in col]
        df["stringency_actual"] = df[stringency_cols].max(axis=1)
        df = df.drop(stringency_cols, axis=1)
        return df
    
    def _preprocess_weather(self, df):
        rename_dict = {
            "DDVEC":"vector_mean_wind_direction_in_degrees",
            "FHVEC":"vector_mean_wind_speed",
            "FG":"24h_average_wind_speed",
            "FHX":"highest_hourly_average_wind_speed",
            "FHXH":"hour_segment_highest_hourly_average_wind_speed",
            "FHN":"lowest_hourly_average_wind_speed",
            "FHNH":"hour_segment_lowest_hourly_average_wind_speed",
            "FXX":"highest_wind_gust",
            "FXXH":"hour_segment_highest_wind_gust",
            "TG":"24h_average_temperature",
            "TN":"minimum_temperature",
            "TNH":"hour_segment_minimum_temperature",
            "TX":"maximum_temperature",
            "TXH":"hour_segment_maximum_temperature",
            "T10N":"minimum_temperature_at_10cm",
            "T10NH":"6h_period_minimum_temperature_at_10cm",
            "SQ":"sunshine_duration",
            "SP":"percentage_of_longest_possible_sunshine_duration",
            "Q":"global_radiation",
            "DR":"duration_of_precipitation",
            "RH":"24h_sum_of_precipitation",
            "RHX":"highest_hourly_sum_of_precipitation",
            "RHXH":"hour_segment_highest_hourly_sum_of_precipitation",
            "PG":"24h_average_air_pressure_converted_to_sea_level",
            "PX":"highest_hourly_value_of_air_pressure_reduced_to_sea_level",
            "PXH":"hourly_division_highest_hourly_value_of_air_pressure",
            "PN":"lowest_hourly_value_of_air_pressure_from_sea_level",
            "PNH":"hourly_division_lowest_hourly_value_of_air_pressure",
            "VVN":"minimum_occurred_visibility",
            "VVNH":"hour_minimum_occurred_visibility",
            "VVX":"maximum_visibility_occurred",
            "VVXH":"hour_section_maximum_visibility_occurred",
            "NG":"24h_average_cloud_cover",
            "UG":"24h_average_relative_humidity",
            "UX":"maximum_relative_humidity",
            "UXH":"hour_segment_maximum_relative_humidity",
            "UN":"minimum_relative_humidity",
            "UNH":"hour_section_minimum_relative_humidity",
            "EV24":"reference_crop_evaporation"
        }
        df = df.rename(columns=rename_dict)
        df["date"] = df["date"].astype("str").apply(lambda x: str(x)[:10])
        df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
        df = df.set_index("date")
        df = df[["24h_average_wind_speed", "24h_average_temperature", "24h_sum_of_precipitation", "24h_average_air_pressure_converted_to_sea_level", "24h_average_cloud_cover", "24h_average_relative_humidity"]]
        df = df.replace(-1, 0)
        df.index = df.index.astype('datetime64[ns]')
        return df
    
    def _preprocess_dates(self, date_col: str = "date"):
        df = pd.DataFrame(index=pd.date_range(start=self.start_date, end=self.end_date, freq="D"))
        df.index.name = date_col
        df["dt_iso_year_week"] = df.index.strftime('%Y-%V') # %V is the actual ISO week, but W starts with 0 in a new year, making more sense as a linear variable
        df["dt_iso_year_week"] = df["dt_iso_year_week"].replace("2021-53", "2020-53")
        df['dt_year'] = df.index.year
        df['dt_month'] = df.index.month
        df["dt_day"] = df.index.day
        df["dt_day_of_week"] = df.index.dayofweek
        df["dt_iso_week"] = df.index.strftime('%W').astype("int64") # %V is the actual ISO week, but W starts with 0 in a new year, making more sense as a linear variable
        # One-hot encode week/weekend
        df["dt_is_weekend"] = df["dt_day_of_week"] < 5
        df["dt_is_week"] = ~df["dt_is_weekend"]
        # One-hot encode weekdays
        weekdays = {0: "monday",
                   1: "tuesday",
                   2: "wednesday",
                   3: "thursday",
                   4: "friday",
                   5: "saturday",
                   6: "sunday"}
        for i, d in weekdays.items():
            df[f"dt_is_{d}"] = (df["dt_day_of_week"] == i).astype("int64")
        df.index = df.index.astype('datetime64[ns]')
        return df
