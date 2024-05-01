import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.lof import LOF
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
import seaborn as sns


class Algo:
    
    #working
    def predictRF(N, P, K, temp, hum, ph, rainfall):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import MinMaxScaler

        # Load the dataset
        csv_data = pd.read_csv('crop_recommendation.csv')
        # Extract the features and label columns
        X = csv_data.drop('label', axis=1).values
        y = csv_data['label'].values

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features using Min-Max Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)

        new_data = [[N, P, K, temp, hum, ph, rainfall]] 
        new_data_scaled = scaler.transform(new_data)
    
        value=rf.predict(new_data_scaled)
        return value


    #working
    def predictDtree(N, P, K, temp, hum, ph, rainfall):
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import MinMaxScaler
        # Load the dataset
        csv_data = pd.read_csv('crop_recommendation.csv')
        # Extract the features and label columns
        X = csv_data.drop('label', axis=1).values
        y = csv_data['label'].values
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    
        # Scale the features using Min-Max Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
       # Train the decision tree classifier
        dtree = DecisionTreeClassifier(random_state=42)
        dtree.fit(X_train_scaled, y_train)

    
        new_data = [[N, P, K, temp, hum, ph, rainfall]]   
        new_data_scaled = scaler.transform(new_data)
        value=dtree.predict(new_data_scaled)
        return value

    #incomplete, have to add plots and parameters.
    def pyODLoc():

        data=pd.read_csv("crop_recommendation.csv")
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

        plt.figure(figsize=(10, 6))
        sns.histplot(train_outlier_scores, bins=50, kde=True, color='blue', alpha=0.5, label='Training Data')
        sns.histplot(test_outlier_scores, bins=50, kde=True, color='red', alpha=0.5, label='Testing Data')
        plt.axvline(np.percentile(train_outlier_scores, 99), color='black', linestyle='dashed', linewidth=2, label='Threshold')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Comparison of Predicted and Actual Test Data')
        plt.legend()
        st.pyplot()




    def main():
        data=pd.read_csv("crop_recommendation.csv")
        print(data)

        #describing the data
        print(data.describe())

        #function call to random forest
        #value=predictRF(N, P, K, temp, hum, ph, rainfall)
        #print(value)

    



    



    #comment

    if __name__ == "__main__":
        main()

