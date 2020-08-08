
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

try:
    from sklearn.neighbors import LocalOutlierFactor

except:
    raise Exception("sklearn.neighbors is not installed.")
    

# FUNCTIONS
def what_is_DSH():

    print("-General Information- \n\n Description: Data Science Helper is a open source library that intends to save time \n and make important data science operations without any effort. \n Library Name: Data Science Helper \n Language: Python \n Version: 1.3.0 \n Field: Data science and Machine Learning \n Purpose: Helping with data science operations and save time. \n Used Libraries: Pandas, Numpy, Matplotlib, Seaborn, Scipy \n Github Link: https://github.com/bayhippo/Data-Science-Helper \n Author: Salih Albayrak \n First Creation Date: 7/31/2020 \n Documentation: https://github.com/bayhippo/Data-Science-Helper/wiki")    

    # what fucntion prints
    """
    -General Information- 
    Description: Data Science Helper is a open source library that intends to save time 
    and make important data science operations without any effort. 
    Library Name: Data Science Helper 
    Language: Python 
    Version: 1.3.0 
    Field: Data science and Machine Learning 
    Purpose: Helping with data science operations and save time. 
    Used Libraries: Pandas, Numpy, Matplotlib, Seaborn, Scipy 
    Github Link: https://github.com/bayhippo/Data-Science-Helper 
    Author: Salih Albayrak 
    First Creation Date: 7/31/2020 
    Documentation: https://github.com/bayhippo/Data-Science-Helper/wiki
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
    
    # checking list features length, if it is not even then draws a one big plot for the last feature
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
    plt.show()
            
    # checking list features length, if it is not even then draws a one big plot for the last feature
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
    

def find_correlated_features(data, features, feature, 
                             threshold = -1, 
                             metric = "pearsonr", 
                             plotting = True, 
                             dropping = False,
                             dropping_threshold = 1):
    correlations = []
    correlated_features = []
    dropped_features = []
    for same in features:
        if same == feature:
            features.remove(feature)
        else:
            continue
    
    metric = metric.lower()
    
    if metric == "pearsonr":
        for other_feature in features:
            correlation = stats.pearsonr(data[feature], data[other_feature])[0]

            if correlation >= threshold:
                correlations.append(correlation)
                correlated_features.append(other_feature)
                
                if dropping == True:
                    if correlation > dropping_threshold:
                        data = data.drop([other_feature], axis = 1)
                        dropped_features.append(other_feature)

            else:
                continue
                
    if metric == "spearmanr":
        for other_feature in features:
            correlation = stats.spearmanr(data[feature], data[other_feature])[0]

            if correlation >= threshold:
                correlations.append(correlation)
                correlated_features.append(other_feature)

            else:
                continue
    
    df = {"features" : correlated_features, "correlations" : correlations} 
    df = pd.DataFrame(df)
    df = df.sort_values(by = "correlations")
                
    if plotting == True:         
        plt.figure(figsize = (15,8))
        sns.barplot(x = df["features"], y = df["correlations"], palette = "plasma_r") 
        plt.title("Correlated Features with {}(Metric: {})".format(feature, metric))
        plt.xticks(rotation = 90)
        plt.axhline(y = 0, color = "black", label = "Zero Line")
        if dropping == True:
            plt.axhline(y = dropping_threshold, color = "red",linestyle = "dashed", label = "Dropping Threshold({})".format(dropping_threshold))
        plt.legend()
        plt.show()
                
    return data, dropped_features
    
    
def outlier_detector(data, features, feature1, feature2, threshold, plotting = True):
    
    x = data[features]
    
    clf = LocalOutlierFactor()
    y_pred = clf.fit_predict(x)
    X_score = clf.negative_outlier_factor_

    outlier_score = pd.DataFrame()
    outlier_score["score"] = X_score
    outlier_score.head()

    filter1 = outlier_score["score"] < threshold
    outlier_index = outlier_score[filter1].index.tolist()
        
    x_len = len(x.drop(outlier_index))

    if plotting == True:
        fig, ax = plt.subplots(1,1, figsize = (13,8))
        plt.scatter(x[feature1],x[feature2],color = "k",s = 6,label = "Data Points")
        f1_index = x.columns.get_loc(feature1)
        f2_index = x.columns.get_loc(feature2)
        plt.scatter(x.iloc[outlier_index,f1_index],x.iloc[outlier_index,f2_index],color = "red",s = 30, label = "Outlier")

        radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
        outlier_score["radius"] = radius
        plt.scatter(x[feature1], x[feature2], s = 1000*radius, edgecolor = "b", facecolors = "none", label = "Outlier Score")
        plt.legend()
        plt.xlabel("{}".format(feature1))
        plt.ylabel("{}".format(feature2))
        plt.grid(True,alpha = 0.4) 
        plt.text(0.66,0.1 , 
                 "Number of Outliers:"+str(len(data) - x_len), 
                 horizontalalignment='left', 
                 verticalalignment='top',
                 transform=ax.transAxes,
                 fontsize = 18, 
                 color = "black")
        plt.title("Outlier Detection Plot")
        plt.show()

    x = x.drop(outlier_index)
    print("Number of Outliers(Number of Dropped Rows): {}".format(len(data) - x_len))
    print("Min Outlier Score: {}".format(np.min(outlier_score["score"])))
    
    return x, outlier_score["score"]
        

def find_numericlike_categorical_features(data, features, filter_feature):
    """
    This function's purpose is to find features that seems like numeric
    but categorical in real. For example number of rooms of a house seems like
    a numeric feature but in reality if you think it as a categorical feature 
    you will probably get better results. For a better explanation check out 
    the Github page of this library: https://github.com/bayhippo/Data-Science-Helper/wiki
    """
    
    length_of_features = len(features)
    fig, ax = plt.subplots(int(length_of_features/2),2, figsize = (12,length_of_features*3)) 
    
    # drwing plots
    count = 0
    for r in range(0,int(length_of_features/2)):
        for c in range(0,2):
            feature = features[count]
            ax[r,c].scatter(data[feature], data[filter_feature])
            count += 1
            ax[r,c].set_title(feature + "-" + filter_feature)
            ax[r,c].set_xlabel(feature)
            ax[r,c].set_ylabel(filter_feature)
            ax[r,c].grid(True, alpha = 0.4)
    plt.show()
            
    # checking list features length, if it is not even then draws a one big plot for the last feature
    plt.show()
    if length_of_features%2 == 1:
        plt.figure(figsize = (12,8))
        plt.scatter(data[features[length_of_features-1]], data[filter_feature])
        plt.title(features[length_of_features-1] + "-" + filter_feature)
        plt.grid(True, alpha = 0.4)
        plt.show()
