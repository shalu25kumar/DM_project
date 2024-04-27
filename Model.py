import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.lof import LOF
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
    

def main():
    data=pd.read_csv("crop_recommendation.csv")
    print(data)

    #describing the data
    print(data.describe())

    # Separating  features and labels
    X = data.iloc[:, :-1].values  # Features: N, P, K, temperature, humidity, ph, rainfall
    y = data.iloc[:, -1].values    # Labels

    print(X)
    print(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LOF model
    lof = LOF(contamination=0.1)  
    lof.fit(X_train_scaled)
    
    train_outlier_scores = lof.decision_function(X_train_scaled)
    test_outlier_scores = lof.decision_function(X_test_scaled)

    print("train_outlier_scores")
    print(train_outlier_scores)

    print("test_outlier_scores")
    print(test_outlier_scores)






    









    



#comment

if __name__ == "__main__":
    main()

