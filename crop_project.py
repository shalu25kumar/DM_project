from statistics import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from pyod.models.lof import LOF
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import requests
from io import BytesIO
from PIL import Image
import os
import requests
import Model

def set_background(background_image_url):
    """
    Function to set background image using HTML and CSS.
    """
    try:
        # Load the image from the URL
        response = requests.get(background_image_url)
        response.raise_for_status()

        # Convert the image to base64 encoding
        image = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Apply HTML and CSS to set background image
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_str}");
            background-size: cover;
        }}
        </style>
        """,
            unsafe_allow_html=True
        )

    except requests.exceptions.RequestException as e:
        st.write(f"Error loading image from URL: {e}")

    except Image.UnidentifiedImageError as e:
        st.write(f"Error identifying image format: {e}")
#

# Display title with custom fontDD
#st.title("CROP RECOMMENDATION")
st.markdown("""
<style>
.custom-font {
    font-family: 'Roboto', sans-serif;
    font-size: 50px;
    font-weight: bold;
    color: #000000;
    text-align: center;
    margin-top: 50px;
    
}
</style>
<div class="custom-font">CROP RECOMMENDATION</div>
""", unsafe_allow_html=True)
#load data (default loading)
#data =pd.read_csv("crop_recommendation.csv")
#st.write(data)

#Load CSV data side option for uploading
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

# Display data and plot
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("##CSV Data")
    #st.dataframe(data)
    st.dataframe(data.style.set_properties(**{'background-color': 'yellow'}))
    #st.write(data, width=10000)


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
    rf = RandomForestClassifier(n_estimators=100, random_state=20)
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


def predictRegression(N, P, K, temp, hum, ph, rainfall):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    # Load the dataset
    csv_data = pd.read_csv('crop_recommendation.csv')
    # Extract the features and label columns
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    
    # Scale the features using Min-Max Scaling
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    #X_test_scaled = scaler.transform(X_test)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Example new data for prediction
    new_data = [[N, P, K, temp, hum, ph, rainfall]]  
    new_data_scaled = scaler.transform(new_data)
    value=model.predict(new_data_scaled)
    value=le.inverse_transform(value.astype(int))
    
    return value

# Example function to define best crop per cluster
def define_best_crops_per_cluster(cluster_labels, csv_data):
    best_crops = {}
    for cluster in set(cluster_labels):
        cluster_crops = csv_data[csv_data['cluster'] == cluster]['label']
        best_crop = cluster_crops.value_counts().idxmax()  # This assumes the most common crop is the best
        best_crops[cluster] = best_crop
    return best_crops


# Function to recommend crops based on input conditions
def recommend_crops(input_temperature, input_rainfall, input_ph, input_humidity, input_N, input_P, input_K, scaler, kmeans, best_crops,csv_data):
    input_data_scaled = scaler.transform([[input_temperature, input_rainfall, input_ph, input_humidity, input_N, input_P, input_K]])
    predicted_cluster = kmeans.predict(input_data_scaled)[0]
    best_crop = best_crops[predicted_cluster]
  
    # Get other crops in the same cluster
    other_crops = csv_data[csv_data['cluster'] == predicted_cluster]['label'].unique().tolist()
    other_crops.remove(best_crop)
  
    return  predicted_cluster,best_crop,other_crops


def Kmeans(N, P, K, temp, hum, ph, rainfall):
    # Load your dataset

    csv_data = pd.read_csv('crop_recommendation.csv')  # Update path accordingly

    k = 5  # Number of clusters
    numeric_data = csv_data[['temperature', 'rainfall', 'ph', 'humidity', 'N', 'P', 'K']]
    scaler = StandardScaler().fit(numeric_data)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaler.transform(numeric_data))
    csv_data['cluster'] = kmeans.labels_
    best_crops = define_best_crops_per_cluster(csv_data['cluster'], csv_data)
    predicted_cluster, best_crop, other_crops = recommend_crops(temp, rainfall, ph, hum, N, P, K, scaler, kmeans,best_crops,csv_data)
    # Define which columns are numeric and which are categorical
    numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    categorical_cols = ['label']  

    # Define preprocessing steps for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Define the KMeans clustering pipeline
    kmeans_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=5, random_state=0))
    ])

    # Fit the pipeline
    kmeans_pipe.fit(csv_data[numeric_cols + categorical_cols])

    # Get cluster labels
    csv_data['cluster'] = kmeans_pipe.named_steps['kmeans'].labels_


    grouped = csv_data.groupby('cluster')['label'].value_counts().unstack(fill_value=0)


    # Normalize the counts to get a score that sums to 1 for comparison
    grouped_norm = grouped.div(grouped.sum(axis=1), axis=0)

    return  best_crop, other_crops





def pyODLoc():
    data=pd.read_csv("crop_recommendation.csv")
    # Separating  features and labels
    X = data.iloc[:, :-1].values  # Features: N, P, K, temperature, humidity, ph, rainfall
    y = data.iloc[:, -1].values    # Labels
    #print(X)
    #print(y)
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
   

     
   # Plot the histograms of the anomaly scores with threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(train_outlier_scores, bins=50, kde=True, color='blue', alpha=0.5, label='Training Data', ax=ax)
    sns.histplot(test_outlier_scores, bins=50, kde=True, color='red', alpha=0.5, label='Testing Data', ax=ax)
    ax.axvline(np.percentile(train_outlier_scores, 99), color='black', linestyle='dashed', linewidth=2, label='Threshold')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title('Comparison of Predicted and Actual Test Data')
    ax.legend()
    st.pyplot(fig)
    st.write(" ")


    fig, ax = plt.subplots(figsize=(25, 10))
    st.markdown("<p style='color:black; font-size:20px; font-weight:bold'>BOX PLOTS</p>", unsafe_allow_html=True)
    # Use boxplot instead of histplot
    sns.boxplot(x=y_test, y=test_outlier_scores, ax=ax)

    # Customize the plot
    ax.set_xlabel('Actual Labels')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Boxplot of Anomaly Scores by Actual Labels')
    st.pyplot(fig)
       

    


def main():
    #background_image_url = "https://cdn.pixabay.com/photo/2024/04/08/14/09/nature-8683570_1280.jpg"
    #background_image_url="https://images.squarespace-cdn.com/content/v1/59765fd317bffcafaf5ff75c/1533928693210-QM2PRAVOH4ZNY13KHLET/ke17ZwdGBToddI8pDm48kD6g6d_8IznzvwGE9lO5DQoUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYy7Mythp_T-mtop-vrsUOmeInPi9iDjx9w8K4ZfjXt2dkohZJiDdrI1I8fvN-mvKNK5lz5E2twVKTvGuJDiNEjuG6v6ULRah83RgHXAWD5lbQ/image-asset.jpeg"
    background_image_url="https://www.martycohenphotography.com/wp-content/uploads/2016/02/Row_Crops_0463-M-MPIX-Test.jpg"
    set_background(background_image_url)
    
    # Six input fields
    st.header("Your Data")
    
    # Create three columns to place buttons side by side
    col1, col2, col3 = st.columns(3)

    # Button 1 in the first column
    with col1:
        inputN = st.text_input("N Value  (should be between 0 and 140)", "")
        inputT = st.text_input("Tempature  (should be between 5 and 45)", "")
        inputR = st.text_input("Rainfall   (should be between 20 and 300)", "")

    # Button 2 in the second column
    with col2:
        inputP = st.text_input("P Value  (should be between 5 and 145)", "")
        inputH = st.text_input("Humidity    (should be between 10 and 100)", "")
        

    # Button 3 in the third column
    with col3:
              inputK = st.text_input("K Value  (should be between 5 and 205)", "")
              inputph = st.text_input("pH Value    (should be between 2 and 10)", "")



              
    st.write(" ")
    st.header("Choose a method :  ")

    coll1, coll2, coll3,coll4 = st.columns(4)

    # Button 1 in the first column
    with coll1:
        if st.button("Random Forest"):
            
            st.write("N value:", inputN,
                 "P value:", inputP, 
                 "K value:", inputK, 
                 "Tempature:",inputT,
                 "Humidity:", inputH,
                 "pH Value:", inputph,
                 "Rainfall:", inputR)
            if not inputN or not inputP or not inputK or not inputT or not inputH or not inputph or not inputR:
                st.write("Please fill in all the input fields.")
                return
            # call the predictRF function here with the input values
            result = predictRF(float(inputN), float(inputP), float(inputK), float(inputT), float(inputH), float(inputph), float(inputR))
            st.write("Recommended crop from Random Forest : ", result)
            
    # Button 2 in the second column
    with coll2:
        if st.button("Decision trees"):
            st.write("N value:", inputN,
                 "P value:", inputP, 
                 "K value:", inputK, 
                 "Tempature:",inputT,
                 "Humidity:", inputH,
                 "pH Value:", inputph,
                 "Rainfall:", inputR)
            if not inputN or not inputP or not inputK or not inputT or not inputH or not inputph or not inputR:
                st.write("Please fill in all the input fields.")
                return
            # call the predictRF function here with the input values
            Dresult = predictDtree(float(inputN), float(inputP), float(inputK), float(inputT), float(inputH), float(inputph), float(inputR))
            st.write("Recommended crop from Decision trees : ", Dresult)

    # Button 3 in the third column
    with coll3:
        if st.button("Rgresssion"):
            st.write("N value:", inputN,
                 "P value:", inputP, 
                 "K value:", inputK, 
                 "Tempature:",inputT,
                 "Humidity:", inputH,
                 "pH Value:", inputph,
                 "Rainfall:", inputR)
            if not inputN or not inputP or not inputK or not inputT or not inputH or not inputph or not inputR:
                st.write("Please fill in all the input fields.")
                return
                # call the predictRF function here with the input values
            Dresult = predictDtree(float(inputN), float(inputP), float(inputK), float(inputT), float(inputH), float(inputph), float(inputR))
            st.write("Recommended crop from Decision trees : ", Dresult)

       
    with coll4:
        if st.button("K_Means Cluster"):
            
            st.write("N value:", inputN,
                 "P value:", inputP, 
                 "K value:", inputK, 
                 "Tempature:",inputT,
                 "Humidity:", inputH,
                 "pH Value:", inputph,
                 "Rainfall:", inputR)
            if not inputN or not inputP or not inputK or not inputT or not inputH or not inputph or not inputR:
                st.write("Please fill in all the input fields.")
                return
            # call the predictRF function here with the input values
            result,suggestedcrops= Kmeans(float(inputN), float(inputP), float(inputK), float(inputT), float(inputH), float(inputph), float(inputR))
            st.write("Recommended crop from  : ", result)
            st.write("suggested crops  : ", suggestedcrops)
   

    




    st.header("Data being used")


    #load data (default loading)
    data = pd.read_csv("crop_recommendation.csv")

    # Define custom CSS
    css = """
        <style>
            .dataframe {
                background-color: transparent !important;
                color: black ;
            }
        </style>
    """

    # Display data with custom CSS
    st.write(css, unsafe_allow_html=True)
    st.dataframe(data.style.set_properties(**{'width': '500px', 'height': '100px'}))

  

    st.write(" ")
    st.header(" ")
    st.header("LOC Algorithm to determine outliers in data")
    st.write("Training and Test Histogram")
    pyODLoc()


#comment

if __name__ == "__main__":
    main()

