import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced
Wellness Tourism Package based on below parameters.
""")

Age = st.number_input("Age", 18.0, 61.0, 40.0)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", ["1", "2","3"])
DurationOfPitch = st.number_input("Duration Of Pitch", 5,127,40)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried","Small Business","Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", 1, 5, 3)
NumberOfFollowups = st.number_input("Number Of Followups", 1, 6, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard","Deluxe","Super Deluxe","King"])
PreferredPropertyStar = st.number_input("Preferred Property Star", 3, 5, 4)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married","Unmarried","Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", 1, 22, 10)
Passport = st.selectbox("Passport", ["0", "1"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Own Car", ["0", "1"])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", 0, 3, 2)
Designation = st.selectbox("Designation", ["VP", "AVP","Manager","Senior Manager","Executive"])
MonthlyIncome = st.number_input("Monthly Income", 1000,98678,5000)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict Customer"):
    prediction = model.predict(input_data)[0]
    result = "Cusomer will purchase package" if prediction == 1 else "Cusomer will not purchase pacakge"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
