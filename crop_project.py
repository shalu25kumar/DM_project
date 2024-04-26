import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display title with custom fontDD
st.title("CROP RECOMMENDATION")

#load data
data =pd.read_csv("crop_recommendation.csv")
st.write(data)

# Load CSV data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

# Display data and plot
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("## CSV Data")
    st.dataframe(data)

def main():
    st.title("Recommendations")

    # Six input fields
    st.header("Your Data")
    
    # Create three columns to place buttons side by side
    col1, col2, col3 = st.columns(3)

    # Button 1 in the first column
    with col1:
        inputN = st.text_input("N Value", "")
        inputT = st.text_input("Tempature", "")
        inputR = st.text_input("Rainfall", "")

    # Button 2 in the second column
    with col2:
        inputP = st.text_input("P Value", "")
        inputH = st.text_input("Humidity", "")
        

    # Button 3 in the third column
    with col3:
              inputK = st.text_input("K Value", "")
              inputph = st.text_input("Ph Value", "")
              
    st.header("Choose")

    coll1, coll2, coll3 = st.columns(3)

    # Button 1 in the first column
    with coll1:
        if st.button("Button 1"):
            st.write("Button 1 clicked!")
            st.write("N value:", inputN,
                     "P value:", inputP, 
                     "K value:", inputK, 
                     "Tempature:",inputT)

    # Button 2 in the second column
    with coll2:
        if st.button("Button 2"):
            st.write("Button 2 clicked!")

    # Button 3 in the third column
    with coll3:
              if st.button("Button 3"):
               st.write("Button 3 clicked!")

#comment

if __name__ == "__main__":
    main()

