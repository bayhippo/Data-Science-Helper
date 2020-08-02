# IMPORTS
try:
    import numpy as np

except:
    raise Exception("numpy is not installed.")
    
    
try:
    import pandas as pd

except:
    raise Exception("pandas is not installed.")
    
    
try:
    import seaborn as sns

except:
    raise Exception("seaborn is not installed.")
    
    
try:
    import matplotlib.pyplot as plt

except:
    raise Exception("matplotlib.pyplot is not installed.")

    
try:
    from scipy import stats

except:
    raise Exception("scipy is not installed.")
    
    
try:
    from scipy.stats import skew

except:
    raise Exception("scipy.stats is not installed.")


# FUNCTIONS
def what_is_DSH():
    print("-General Information- \n\n Description: Data Science Helper is a open source library that intends to save time \n and make important data science operations without any effort. \n Library Name: Data Science Helper \n Language: Python \n Version: 1.0.0 \n Field: Data science and Machine Learning \n Purpose: Helping with data science operations and save time. \n Used Libraries: Pandas, Numpy, Matplotlib, Seaborn, Scipy \n Github Link: https://github.com/bayhippo/Data-Science-Helper \n Author: Salih Albayrak \n First Creation Date: 7/31/2020 \n\n-Function Descriptions- \n\n what_is_DSH(): Explains what Data Science Helper library is. \n Parameters: No parameters, Return: No return \n\n nan_value_vis_and_dropping(): Visualizes NaN value percentages of columns and \n if user wants, it can drop columns according to a threshold. \n Parameters: \n data: Which data fucntion is going to use, \n Value Type: Pandas dataframe \n features: List of data features, Value Type: List, \n threshold_for_dropping: NaN values percentage threshold for dropping a column, \n if a column's NaN values percentage is bigger than that, fuction drops that column, Current Value: 40, Value Type: Float \n dropping: Dropping columns that have NaN values percentages bigger than threshold, \n if user sets this parameter to 'False' then it disables all dropping operations, Current value: True, Value Type: Boolean,\n Return: Data, if user sets dropping to 'True' then funct,on returns data and droped columns. \n\n fill_nan_categorical(): Fills NaN values of categorical columns with column's most frequent value \n and prints information about the feature. \n Parameters: \n data: Which data fucntion is going to use, Value Type: Pandas dataframe \n features: List of data features, Value Type: List \n printing: Prints information about feature, if user sets this parameter to 'False' \n then it disables printings, Current value: True, Value Type: Boolean, Return: Data \n\n fill_nan_numeric(): Fills NaN values of numeric columns with column's mean value and prints information about the feature. \n Parameters: \n data: Which data fucntion is going to use, Value Type: Pandas dataframe \n features: List of data features, Value Type: List \n printing: Prints information about feature, if user sets this parameter to 'False' \n then it disables printings, Current value: True, Value Type: Boolean, Return: Data \n\n show_kdeplot(): Shows kernel denstiy estimation plots of numeric features, \n if feature has values that are less than 1 then it uses distplot. \n Parameters: \n data: Which data fucntion is going to use, Value Type: Pandas dataframe \n features: List of data features, Value Type: List, Return: No return \n\n show_boxplot(): Shows boxplots of numeric features. \n Parameters: \n data: Which data fucntion is going to use, Value Type: Pandas dataframe \n features: List of data features, Value Type: List, Return: Data \n\n boxcox_skewed_data(): Applies 'boxcox' transformation to skewed data according to a threshold. \n Parameters: \n data: Which data fucntion is going to use, Value Type: Pandas dataframe \n features: List of data features, Value Type: List \n threshold: Skewness threshold for boxcox a feature, if a feature's skewnes is bigger than that, fuction drops that feature, \n if feature contains values that are less than 1 then function skips them, \n Current Value: 1.9, Value Type: Float, Return: Data \n\n show_sklearn_model_results(): Draws a lineplot and shows results of a machine learning model's scores \n according to a model's parameter values. \n Parameters: \n test_scores: List of test scores of model, Value Type: List \n train_scores: List of train scores of model, Value Type: List \n parameter_values: List of parameter values of model, Value Type: List \n parameter_name: Name of parameter that used in model, mandatory for printings, Value Type: String \n print_best: Shows information about best parameter value for model, if user sets this parameter to 'False' \n then it disables best model operations, Current Value: True, Value Type: Boolean \n printing: Prints information about parameter values and scores, if user sets this parameter to 'False' \n then it disables printing operations, Current Value: True, Value Type: Boolean \n figsize: Adjusts figsize of plot, Current Value: (15,6), Value Type: Tuple, Return: Best parameter value")
    
    # what fucntion prints
    """
    -General Information- 

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
    """
    

def nan_value_vis_and_dropping(data, features, threshold_for_dropping = 40, dropping = True):
    
    # getting nan percentages of features
    nan_percentages = []
    feature_names = []
    for feature in features:
        if data[feature].isna().sum() > 0:
            nan_percentages.append((data[feature].isna().sum()/len(data))*100)
            feature_names.append(feature)
        else:
            continue
    
    # turning nan percentages and feature names into pandas dataframe
    df_old = {"feature_names":feature_names, "nan_percentages":nan_percentages}
    df = pd.DataFrame(df_old)
    
    df = df.sort_values(by = "nan_percentages")
        
    plt.figure(figsize = (8,15))
    sns.barplot(x = df["nan_percentages"], y = df["feature_names"])
    plt.axvline(threshold_for_dropping, 0,2, color = "black", label = "Dropping Threshold")
    plt.title("Nan Percentages of Features", fontsize=16)
    plt.legend()
    plt.show()
    
    # checking dropping parameter
    if dropping == True:
        df_high_nan_percentage = df[df["nan_percentages"] > threshold_for_dropping]

        print("Dropped columns:")
        for feature_high_nan_percentage in df_high_nan_percentage["feature_names"]:
            data = data.drop([feature_high_nan_percentage], axis = 1)
            current_feature = df_high_nan_percentage[df_high_nan_percentage["feature_names"] == feature_high_nan_percentage]
            print(feature_high_nan_percentage + " dropped" + "({}% Nan)".format(np.round(current_feature["nan_percentages"].values[0],2)))
        
        return data, list(df_high_nan_percentage["feature_names"])
    
    else:
        
        return data
    
    
def fill_nan_categorical(data, features, printing = True):
    
    # finding and filling nan values 
    filled_features = []
    for feature in features:
        if data[feature].isnull().sum().sum() != 0:
            filled_features.append(feature)
            most_frequent = data[feature].value_counts()[:1].sort_values(ascending=False)
            data[feature].fillna(most_frequent.keys()[0],inplace = True)
            #printing
            if printing == True:
                print("Feature: {} \n".format(feature), most_frequent)
                print(data[feature].value_counts())
    
        else:
            continue
    
    # checking printing parameter
    if printing == True:
        print("Filled Features: \n {}".format(filled_features))
        
    return data
    

def fill_nan_numeric(data, features, printing = True):
    
    # finding and filling nan values 
    filled_features = []
    for feature in features:
        if data[feature].isnull().sum().sum() != 0:
            filled_features.append(feature)
            feature_mean = data[feature].mean()
            data[feature].fillna(feature_mean,inplace = True)
            #printing
            if printing == True:
                print("\nFeature: {}".format(feature))
                print("Feature Mean: ",feature_mean, "\n")
                df = pd.DataFrame(data = {feature:data[feature]})
                df.info()
            
        else:
            continue
        
    # checking printing parameter
    if printing == True:
        print("\nFilled Features: \n {}".format(filled_features))
        
    return data


def show_kdeplot(data, features):
    
    # creating figure
    length_of_features = len(features)
    fig, ax = plt.subplots(int(length_of_features/2),2, figsize = (12,length_of_features*3)) 
    
    # drwing plots
    count = 0
    for r in range(0,int(length_of_features/2)):
        for c in range(0,2):
            feature = features[count]
            try:
                sns.kdeplot(data[feature], shade = True, ax = ax[r,c])
            except RuntimeError as re:
                if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
                    sns.distplot(data[feature], kde_kws={'bw': 0.1}, ax = ax[r,c])
                else:
                    raise re
            count += 1
            ax[r,c].set_title(feature + " kdeplot")
    plt.show()
    
    # checking list features length, if it is not even then draws a big 1 plot for the last feature
    if length_of_features%2 == 1:
        plt.figure(figsize = (12,8))
        sns.kdeplot(data[features[length_of_features-1]], shade = True)
        plt.title(features[length_of_features-1] + " kdeplot")
        plt.show()
        
        
def show_boxplot(data, features):
    
    # finding and filling nan values 
    length_of_features = len(features)
    fig, ax = plt.subplots(int(length_of_features/2),2, figsize = (12,length_of_features*3)) 
    
    # drwing plots
    count = 0
    for r in range(0,int(length_of_features/2)):
        for c in range(0,2):
            feature = features[count]
            sns.boxplot(data[feature], ax = ax[r,c])
            count += 1
            ax[r,c].set_title(feature + " boxplot")
            
    # checking list features length, if it is not even then draws a big 1 plot for the last feature
    plt.show()
    if length_of_features%2 == 1:
        plt.figure(figsize = (12,8))
        sns.boxplot(data[features[length_of_features-1]])
        plt.title(features[length_of_features-1] + " boxplot")
        plt.show()
        
        
def boxcox_skewed_data(data, features, threshold = 1.9):
    
    # finding skewed features
    skewed_features = []
    for feature in features:
        non_positive_values = []
        for i in data[feature]: # checking values that are less than 1
            if i < 1:
                non_positive_values.append(i)
        if skew(data[feature]) > threshold and len(non_positive_values) == 0:                
            skewed_features.append(feature)
            boxcoxed, _ = stats.boxcox(data[feature]) # applying boxcox to skewed features using scipy
            data[feature] = boxcoxed
            plt.figure(figsize = (12,8))
            sns.kdeplot(data[feature], shade = True)
            plt.title(feature + " fixed(boxcox)")
            plt.show()
            print("New skew: {}".format(skew(data[feature])))
            
        else:
            continue
        
    # printing skewed features
    print("Skewed Features:")
    print(skewed_features)
    
    return data


def show_sklearn_model_results(test_scores, 
                               train_scores, 
                               parameter_values, 
                               parameter_name, 
                               print_best = True, printing = True,
                               figsize = (15, 6)):
    
    # checking printing parameter and printing results for each parameter value
    if printing == True:
        for index, value in enumerate(parameter_values, start = 0):
            print("=====Parameter Value({}): ".format(parameter_name)+str(value)+"=====")
            print("Test Accuracy: {}/Parameter Value({}): {}".format(np.round(test_scores[index],3), parameter_name, value))
            print("Train Accuracy: {}/Parameter Value({}): {}".format(np.round(train_scores[index],3), parameter_name, value))
    
    # defining variables for finding best parameter value
    parameter_values_start_value = parameter_values[0]
    parameter_values_accrual = np.absolute(parameter_values[1] - parameter_values[0])
    difference_btw_start_accrual = np.absolute(parameter_values_start_value - parameter_values_accrual)
    
    
    # finding best parameter value
    if parameter_values_accrual > parameter_values_start_value:
        best_parameter_value = (parameter_values_accrual*(1+test_scores.index(np.max(test_scores)))) - difference_btw_start_accrual
    
    elif parameter_values_accrual < parameter_values_start_value:
        best_parameter_value = difference_btw_start_accrual + parameter_values_accrual*(1+test_scores.index(np.max(test_scores)))
        
    elif parameter_values_accrual == parameter_values_start_value:
        best_parameter_value = parameter_values_accrual*(1+test_scores.index(np.max(test_scores)))
    
    #plotting
    plt.figure(figsize = figsize)  
    plt.plot(parameter_values,test_scores, label = "Test Accuracy")
    plt.plot(parameter_values,train_scores, c = "orange", label = "Train Accuracy")
    plt.xlabel("Parameter Values({})".format(parameter_name))
    plt.ylabel("Accuracy")
    plt.title("Scores For Each Parameter Value({})".format(parameter_name),fontsize = 12)
    if print_best == True:
        plt.axvline(best_parameter_value, 0,2, color = "black", label = "Best Parameters({})".format(best_parameter_value))
        plt.axhline(np.max(test_scores) ,best_parameter_value, 0, color = "red", linestyle = "dashed", alpha = 0.5)
        plt.axhline(np.max(train_scores) ,best_parameter_value, 0, color = "red", linestyle = "dashed", alpha = 0.5)
    plt.grid(True , alpha = 0.4)
    plt.legend()
    plt.show()

    # printing best parameter value
    if print_best == True and printing == True:
        print("Best Accuracy(test): {}/Parameter Value({}): {}".format(np.round(np.max(test_scores),3), parameter_name, best_parameter_value))
    
    return best_parameter_value
    
