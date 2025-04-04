import pandas as pd
import numpy as np
import torch
import pickle as pkl
import os
from dateutil import tz

utc = tz.tzutc()


def preprocess(dataset_dir, training_year_list, val_year_list, testing_year_list, month_list, window_size):

    X_train = list()
    Y_train = list()

    X_val = list()
    Y_val = list()

    X_test = list()
    Y_test = list()

    num_valid_locations = 0
    num_hours_per_location = []

    df_dir = os.path.join(dataset_dir, 'df')  # the directory where the dataframes are stored

    for file_name in os.listdir(df_dir):   # open a specific location
        with open(os.path.join(df_dir, file_name), 'rb') as f:
            d = pd.read_parquet(f)
        

        if len(d.columns) == 46:  # A valid location: Because 23/261 counties has 47 columns, then we set locations having 46 columns as valid locations
            num_valid_locations += 1
            num_hours_per_location.append(len(d.index))

            X_train_hourly = list()
            Y_train_hourly = list()
            X_val_hourly = list()
            Y_val_hourly = list()
            X_test_hourly = list()
            Y_test_hourly = list()

            X_train_daily = list()
            Y_train_daily = list()
            X_val_daily = list()
            Y_val_daily = list()
            X_test_daily = list()
            Y_test_daily = list()

            train_hour_count = 0  # for each hour in that specific location
            val_hour_count = 0
            test_hour_count = 0

            for index, row in d.iterrows():  # for each hour in that specific location

                if in_period_of(index, training_year_list, month_list):

                    # weather features
                    feature_items = torch.tensor(row.values[:len(row) - 1]).double()
                    feature_items = torch.nan_to_num(feature_items, nan=0.0)

                    # extreme weather (i.e., thunderstorm) labels, 0 means not happen
                    label_items = torch.unsqueeze(torch.tensor(row.values[len(row) - 1]).double(), 0)

                    X_train_hourly.append(feature_items)
                    Y_train_hourly.append(label_items)

                    train_hour_count += 1
                    if train_hour_count % window_size == 0:
                        X_train_daily.append(torch.stack(X_train_hourly))
                        Y_train_daily.append(torch.stack(Y_train_hourly))
                        X_train_hourly = list()
                        Y_train_hourly = list()

                if in_period_of(index, val_year_list, month_list):
                    feature_items = torch.tensor(row.values[:len(row) - 1]).double()
                    feature_items = torch.nan_to_num(feature_items, nan=0.0)
                    label_items = torch.unsqueeze(torch.tensor(row.values[len(row) - 1]).double(), 0)

                    X_val_hourly.append(feature_items)
                    Y_val_hourly.append(label_items)

                    val_hour_count += 1
                    if val_hour_count % window_size == 0:
                        X_val_daily.append(torch.stack(X_val_hourly))
                        Y_val_daily.append(torch.stack(Y_val_hourly))
                        X_val_hourly = list()
                        Y_val_hourly = list()

                if in_period_of(index, testing_year_list, month_list):
                    feature_items = torch.tensor(row.values[:len(row) - 1]).double()
                    feature_items = torch.nan_to_num(feature_items, nan=0.0)
                    label_items = torch.unsqueeze(torch.tensor(row.values[len(row) - 1]).double(), 0)

                    X_test_hourly.append(feature_items)
                    Y_test_hourly.append(label_items)

                    test_hour_count += 1
                    if test_hour_count % 24 == 0:
                        X_test_daily.append(torch.stack(X_test_hourly))
                        Y_test_daily.append(torch.stack(Y_test_hourly))

                        X_test_hourly = list()
                        Y_test_hourly = list()

            X_train.append(torch.stack(X_train_daily))
            Y_train.append(torch.stack(Y_train_daily))
            X_val.append(torch.stack(X_val_daily))
            Y_val.append(torch.stack(Y_val_daily))
            X_test.append(torch.stack(X_test_daily))
            Y_test.append(torch.stack(Y_test_daily))
            print("Time-series from " + file_name + " is extracted.")

    print("\nValid locations: " + str(num_valid_locations))
    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    X_val = torch.stack(X_val)
    Y_val = torch.stack(Y_val)
    X_test = torch.stack(X_test)
    Y_test = torch.stack(Y_test)

    print('\nShape format: (num_counties, num_days, window_size (e.g., num_hours_a_day), num_features)')
    print('Training shape: {}||{}, validation shape: {}||{}, testing shape: {}||{}'.format(X_train.shape,
                                                                                              Y_train.shape,
                                                                                              X_val.shape,
                                                                                              Y_val.shape,
                                                                                              X_test.shape,
                                                                                              Y_test.shape))

    if len(set(num_hours_per_location)) == 1:
        print("\nAll counties share the same length of hourly data")
    else:
        print("\nTime dimension for each location is not consistent, may disable the DAG learning for each timestamp. Please check your data.")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def lastday_in_month(year, month):
    if year % 4 != 0:  # not leap year
        lastday = {1: 31, 3: 31, 5: 31, 7: 31, 8: 31, 10: 31, 12: 31, 4: 30, 6: 30, 9: 30, 11: 30, 2: 28}
    else:
        lastday = {1: 31, 3: 31, 5: 31, 7: 31, 8: 31, 10: 31, 12: 31, 4: 30, 6: 30, 9: 30, 11: 30, 2: 29}
    return lastday[month]


def in_period_of(index, year_list, month_list):
    flag = False
    for year in year_list:
        for month in month_list:
            if pd.Timestamp(year, month, 1, 0).replace(tzinfo=utc) <= index <= pd.Timestamp(year, month, lastday_in_month(year, month), 23).replace(tzinfo=utc):
                flag = True
                break
        if flag == True:
            break
    return flag


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.dirname(current_dir)
    project_root_dir = os.path.dirname(task_dir)

    input_data_dir = os.path.join(project_root_dir, 'Data' , 'ClimateBench-M-TS')

    # where you want your preprocessed store, for example
    output_data_dir = os.path.join(current_dir, 'processed_weather_data')

    # cross-validation: leave one year out
    year_dict = {1: [[2018, 2019, 2020], [2021], [2017]],
                 2: [[2019, 2020, 2021], [2017], [2018]],
                 3: [[2020, 2021, 2017], [2018], [2019]],
                 4: [[2021, 2017, 2018], [2019], [2020]]}

    # controlled group: months for the high frequent thunderstrom season
    season_dict = {0: [5, 6, 7, 8]}

    # take a length of window sized time series, and forecast the next window sized time series
    window_size = 24

    for i in season_dict.keys():
        for j in year_dict.keys():
            print('controlled group {}, cross_validation {}'.format(i, j))
            X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess(input_data_dir,
                                                                        training_year_list = year_dict[j][0],
                                                                        val_year_list = year_dict[j][1],
                                                                        testing_year_list = year_dict[j][2],
                                                                        month_list = season_dict[i],
                                                                        window_size = window_size)

            torch.save(X_train, output_data_dir + '/' + str(i) + '_' + str(j) + '_' + 'X_train.pt')
            torch.save(Y_train, output_data_dir + '/' + str(i) + '_' + str(j) + '_' + 'Y_train.pt')
            torch.save(X_val, output_data_dir + '/' + str(i) + '_' + str(j) + '_' + 'X_val.pt')
            torch.save(Y_val, output_data_dir + '/' + str(i) + '_' + str(j) + '_' + 'Y_val.pt')
            torch.save(X_test, output_data_dir + '/' + str(i) + '_' + str(j) + '_' + 'X_test.pt')
            torch.save(Y_test, output_data_dir + '/' + str(i) + '_' + str(j) + '_' + 'Y_test.pt')

