import pandas as pd
import sys
import numpy as np

sys.path.insert(1,"common")
from constants import *


def predict_default(loaded_model, data_file, model_scaler):
    aggregated_column_list = []
    new_cat_names = []
    aggregated_numerical_features = []

    non_default = data_file.loc[data_file['default'].isnull()]
    print non_default.shape
    non_default.drop(columns=dropped_columns_due_to_missing_data, axis=1, inplace=True)
    for key in aggregation_dict:
        # print aggregation_dict[key]
        aggregated_numerical_features.append(key)
        aggregator = 0
        temp_df = pd.DataFrame()
        for col,factor in aggregation_dict[key].items():
            aggregator += factor
            aggregated_column_list.append(col)
            temp_df = pd.concat([temp_df, factor*non_default[col]], axis=1)
        non_default[key] = pd.Series(temp_df.sum(axis=1)/aggregator)
    print non_default.shape
    non_default.drop(columns=aggregated_column_list, axis=1, inplace=True)
    print non_default.columns
    print non_default.shape
    numX = [col for col in numerical_parameters if col in non_default.columns]
    numX += aggregated_numerical_features
    print "***********************"
    print numX
    numericX = non_default[numX]

    print numericX.columns
    scaled_numericX = pd.DataFrame(model_scaler.transform(numericX), columns=numericX.columns)

    catX = pd.DataFrame()
    categorical_features = [feature for feature in categorical_parameters if feature not in dropped_columns_due_to_missing_data]
    for col in categorical_features:
        cat_col = col + "_cat"
        new_cat_names.append(cat_col)
        cat_col = pd.get_dummies(pd.Categorical(pd.Series(non_default[col])), prefix=col, drop_first=True)
        catX = pd.concat([catX, cat_col], axis=1)

    final_data = pd.concat([scaled_numericX, catX], axis=1)
    print "final_data : ", final_data.shape
    pmi_drop_column_list_for_input_data = [col for col in final_data.columns if col in pmi_drop_column_list]
    final_data.drop(list(pmi_drop_column_list_for_input_data), axis=1, inplace=True)
    print final_data.columns
    y_pred = loaded_model.predict(final_data)
    y_pred = pd.DataFrame(pd.Series(np.where(y_pred==-1, 0, y_pred)))

    output_df = pd.concat([non_default['uuid'], y_pred], axis=1)
    return output_df