import streamlit as st
import pandas as pd
import pickle

# Load the actual trained model
with open("model_gb.pkl", "rb") as model_file:
    model = pickle.load(model_file)

#load the actual classifier
with open("model_nn.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)  
r=st.sidebar.radio("Select Page", ("Home", "Predict Revenue", "Predict Classification"))
if r=="Home":
    st.title("Welcome to Clickstream Revenue Prediction App")
    st.write("This app allows you to predict revenue based on clickstream data.")
    
    if st.button("Go to Predict Revenue"):
            st.write("Please navigate to the 'Predict Revenue' page to upload your data and get predictions.")
            st.write("GO TO PREDICT REVENUE PAGE in the sidebar to upload your data and get predictions.")
    if st.button("Go to Predict Classification"):
        st.write("You can navigate to the 'Predict Classification' page to upload your data and get classifications.")
        st.write("GO TO PREDICT CLASSIFICATION PAGE in the sidebar to upload your data and get classifications.")
# Streamlit app for clickstream revenue prediction
if r=="Predict Revenue":
    st.title("Clickstream Revenue Prediction")


    uploaded_file = st.file_uploader("C:/Users/murkr/Downloads/test_data_cleaned.csv", type="csv")

    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(test_data.head())

        try:
        # Select only the features used during training
            selected_features = ["month", "country", "page1_main_category", "colour", "location", "model_photography", "page2_clothing_model", "page", "price_2"]  
            X_test = test_data[selected_features]

            predictions = model.predict(X_test)
            test_data['Predicted_Revenue'] = predictions

            st.write("Predicted Revenue:")
            st.dataframe(test_data[['Predicted_Revenue']])
            st.write("Full Data with Predictions:")
            st.dataframe(test_data)
            #sum the predicted revenue if the classification is 1
            total_revenue = test_data[test_data['price_2'] == 1]['Predicted_Revenue'].sum()
            st.write(f"Total Revenue: {round(total_revenue,2)}")

            st.download_button("Download CSV", test_data.to_csv(index=False), "predicted_output.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
if r=="Predict Classification":
    st.title("Clickstream Classification Prediction")

    uploaded_file = st.file_uploader("C:/Users/murkr/Downloads/test_data_cleaned.csv", type="csv")

    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(test_data.head())

        try:
            # Select only the features used during training
            selected_features = ["month", "country", "page1_main_category", "colour", "location", "model_photography", "page2_clothing_model", "page", "price"]  
            X_test = test_data[selected_features]

            classifications = classifier.predict(X_test)
            test_data['Predicted_Classification'] = classifications
           

            st.write("Predicted Classification:")
            st.dataframe(test_data[['Predicted_Classification']])
            st.write("Full Data with Predictions:")
            st.dataframe(test_data)
            

            st.download_button("Download CSV", test_data.to_csv(index=False), "predicted_classification_output.csv")
        except Exception as e:
            st.error(f"classification failed: {e}")

