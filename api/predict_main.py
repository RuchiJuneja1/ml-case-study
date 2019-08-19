import pandas as pd
import sys
import numpy as np

sys.path.insert(0,"common")
from constants import *


def send_error_file(error_message):
	'''

	:param error_message: error message to be embedded in response csv file
	:return: error dataframe
	'''
	error_df = pd.DataFrame(columns=['error_message'])
	error_df.loc[0] = error_message
	return error_df


def validate_input(data_file):
	'''

	:param data_file: user input data file
	:return: error dataframe if validation fails, "validated" otherwise
	'''
	input_file_column_names = data_file.columns
	if set(input_file_column_names) != set(required_input_file_parameter):
		error_message = "parameters name in given input file does not match the requirement."
		return send_error_file(error_message)

	if len(data_file.loc[data_file['default'].isnull()])==0:
		error_message = "input file does not contain records where default is not defined."
		return send_error_file(error_message)

	return "validated"


def features_aggregation(null_default):
    '''

    :param null_default: pandas dataframe where default is not defined
    :return: altered dataframe with new aggregated fields added
    '''
    aggregated_numerical_features =[]
    aggregated_column_list = []
    for key in aggregation_dict:
        aggregated_numerical_features.append(key)
        aggregator = 0
        temp_df = pd.DataFrame()
        for col,factor in aggregation_dict[key].items():
            aggregator += factor
            aggregated_column_list.append(col)
            temp_df = pd.concat([temp_df, factor*null_default[col]], axis=1)
        null_default[key] = pd.Series(temp_df.sum(axis=1)/aggregator)

    null_default.drop(columns=aggregated_column_list, axis=1, inplace=True)
    return (null_default, aggregated_numerical_features)


def predict_default(loaded_model, data_file, model_scaler):
    '''

    :param loaded_model: pre-trained prediction model file
    :param data_file: user uploaded dataset file
    :param model_scaler: scaler used in the model to scale numeric data
    :return: csv file for the records where default is null. CSV format: <uuid>;<pd>
    '''

    new_cat_names = []

    input_file_validation_status = validate_input(data_file)
    if type(input_file_validation_status)!=str:
		return input_file_validation_status

    # extract records where default is not defined
    null_default = data_file.loc[data_file['default'].isnull()]

    # remove columns that are not required for prediction
    redundant_fields = dropped_columns_due_to_missing_data + correlated_drop_feature_list
    null_default.drop(columns=redundant_fields, axis=1, inplace=True)

    # create new aggregated columns
    null_default_df, aggregated_numerical_features = features_aggregation(null_default)
    numX = [col for col in numerical_parameters if col in null_default_df.columns]
    numX += aggregated_numerical_features
    numericX = null_default_df[numX]
    scaled_numericX = pd.DataFrame(model_scaler.transform(numericX), columns=numericX.columns)

    # create dummy variables for categorical data
    catX = pd.DataFrame()
    categorical_features = [feature for feature in categorical_parameters if feature not in redundant_fields]
    for col in categorical_features:
        cat_col = col + "_cat"
        new_cat_names.append(cat_col)
        cat_col = pd.get_dummies(pd.Categorical(pd.Series(null_default_df[col])), prefix=col, drop_first=True)
        catX = pd.concat([catX, cat_col], axis=1)

    # collate numerical and categorical features
    # remove all features not required for prediction post feature encoding
    final_data = pd.concat([scaled_numericX, catX], axis=1)
    pmi_drop_column_list_for_input_data = [col for col in final_data.columns if col in pmi_drop_column_list]
    low_relevant_features = [col for col in final_data.columns if col in non_important_features]
    features_drop_list = pmi_drop_column_list_for_input_data + low_relevant_features
    final_data.drop(list(features_drop_list), axis=1, inplace=True)

    ordered_final_data = final_data[model_column_list]
    default_pred = loaded_model.predict_proba(pd.DataFrame(ordered_final_data))
    default_pred = np.around(default_pred, decimals=2)
    default_pred = pd.DataFrame(pd.Series([default_1 for default_0, default_1 in default_pred]))

    output_df = pd.concat([pd.DataFrame(null_default_df['uuid']).set_index(default_pred.index), default_pred], axis=1)
    output_df.columns = ['uuid', 'pd']
    return output_df
