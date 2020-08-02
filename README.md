# Data Science Helper
Data Science Helper is a open source library that intends to save time and make important data science operations with less effort. 

# General Information 
Description: Data Science Helper is a open source library that intends to save time 
and make important data science operations without any effort. 
Library Name: Data Science Helper 
Language: Python 
Version: 1.0.0 
Field: Data science and Machine Learning 
Purpose: Helping with data science operations and save time. 
Used Libraries: Pandas, Numpy, Matplotlib, Seaborn, Scipy 
Github Link: https://github.com/bayhippo/Data-Science-Helper 
Author: Salih Albayrak 
First Creation Date: 7/31/2020 

-Function Descriptions- 

what_is_DSH(): Explains what Data Science Helper library is. 
Parameters: No parameters, Return: No return 

nan_value_vis_and_dropping(): Visualizes NaN value percentages of columns and 
if user wants, it can drop columns according to a threshold. 
Parameters: 
data: Which data fucntion is going to use, 
Value Type: Pandas dataframe 
features: List of data features, Value Type: List, 
threshold_for_dropping: NaN values percentage threshold for dropping a column, 
if a column's NaN values percentage is bigger than that, fuction drops that column, Current Value: 40, Value Type: Float 
dropping: Dropping columns that have NaN values percentages bigger than threshold, 
if user sets this parameter to 'False' then it disables all dropping operations, Current value: True, Value Type: Boolean,
Return: Data, if user sets dropping to 'True' then funct,on returns data and droped columns. 

fill_nan_categorical(): Fills NaN values of categorical columns with column's most frequent value 
and prints information about the feature. 
Parameters: 
data: Which data fucntion is going to use, Value Type: Pandas dataframe 
features: List of data features, Value Type: List 
printing: Prints information about feature, if user sets this parameter to 'False' 
then it disables printings, Current value: True, Value Type: Boolean, Return: Data 

fill_nan_numeric(): Fills NaN values of numeric columns with column's mean value and prints information about the feature. 
Parameters: 
data: Which data fucntion is going to use, Value Type: Pandas dataframe 
features: List of data features, Value Type: List 
printing: Prints information about feature, if user sets this parameter to 'False' 
then it disables printings, Current value: True, Value Type: Boolean, Return: Data 

show_kdeplot(): Shows kernel denstiy estimation plots of numeric features, 
if feature has values that are less than 1 then it uses distplot. 
Parameters: 
data: Which data fucntion is going to use, Value Type: Pandas dataframe 
features: List of data features, Value Type: List, Return: No return 

show_boxplot(): Shows boxplots of numeric features. 
Parameters: 
data: Which data fucntion is going to use, Value Type: Pandas dataframe 
features: List of data features, Value Type: List, Return: Data 

boxcox_skewed_data(): Applies 'boxcox' transformation to skewed data according to a threshold. 
Parameters: 
data: Which data fucntion is going to use, Value Type: Pandas dataframe 
features: List of data features, Value Type: List 
threshold: Skewness threshold for boxcox a feature, if a feature's skewnes is bigger than that, fuction drops that feature, 
if feature contains values that are less than 1 then function skips them, 
Current Value: 1.9, Value Type: Float, Return: Data 

show_sklearn_model_results(): Draws a lineplot and shows results of a machine learning model's scores 
according to a model's parameter values. 
Parameters: 
test_scores: List of test scores of model, Value Type: List 
train_scores: List of train scores of model, Value Type: List 
parameter_values: List of parameter values of model, Value Type: List 
parameter_name: Name of parameter that used in model, mandatory for printings, Value Type: String 
print_best: Shows information about best parameter value for model, if user sets this parameter to 'False' 
then it disables best model operations, Current Value: True, Value Type: Boolean 
printing: Prints information about parameter values and scores, if user sets this parameter to 'False' 
then it disables printing operations, Current Value: True, Value Type: Boolean 
figsize: Adjusts figsize of plot, Current Value: (15,6), Value Type: Tuple, Return: Best parameter value
