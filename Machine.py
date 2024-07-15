import pickle
import streamlit as st
import numpy as np
import pandas as pd
import json
from streamlit_lottie import st_lottie
import plotly.express as px

col1 , col2 = st.columns(2)
with col1:
    @st.cache_data(ttl=60 * 60)
    def load_lottie_file(filepath : str):
        with open(filepath, "r") as f:
            gif = json.load(f)
        return gif

    gif = load_lottie_file(r"D:\internships\Data analyst intern 360 digi\project\Robo.json")

    # Display the animation with a placeholder image while loading
    with st.spinner("Loading animation..."):
        st_lottie(gif, speed=1, width=250, height=250)

with col2:
    @st.cache_data(ttl=60 * 60)
    def load_lottie_file(filepath : str):
        with open(filepath, "r") as f:
            gif = json.load(f)
        return gif

    gif = load_lottie_file(r"D:\internships\Data analyst intern 360 digi\project\machine.json")

    # Display the animation with a placeholder image while loading
    with st.spinner("Loading animation..."):
        st_lottie(gif, speed=1, width=250, height=250)
st.title("Fuel pump Machine downtime prediction app")
df = pd.read_csv(r"D:\internships\Data analyst intern 360 digi\project\New_modified_machine_downtime_dataset.csv")


# Load the OneHotEncoder
with open(r'D:\internships\Data analyst intern 360 digi\project\One_hot.pkl', 'rb') as f:
    OH = pickle.load(f)

# Load the RandomForest model
with open(r'D:\internships\Data analyst intern 360 digi\project\Machine_down_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the StandardScaler
with open(r'D:\internships\Data analyst intern 360 digi\project\Machine_scaler.pkl', 'rb') as f:
    ss = pickle.load(f)

with st.sidebar:

    @st.cache_data(ttl=60 * 60)
    def load_lottie_file(filepath : str):
        with open(filepath, "r") as f:
            gif = json.load(f)
        return gif

    gif = load_lottie_file(r"D:\internships\Data analyst intern 360 digi\project\fuel.json")

    # Display the animation with a placeholder image while loading
    with st.spinner("Loading animation..."):
        st_lottie(gif, speed=1, width=350, height=350)

    Month = st.slider("Please choose the month of production: ",min_value=1 , max_value=31)

    hydraulic_pressure = st.text_input("Enter hydraulic pressure bars : " , key = 1)
    st.warning("**Note :** For optimal results, input values between 150 - 300 bars or any number of bars.")

    coolant_pressure = st.text_input("Enter coolant pressure bars : ", key = 2)
    st.warning("**Note :** For optimal results, input values between 1 - 1.5 bars or any number of bars.")

    air_pressure = st.text_input("Enter air pressure bars : " , key = 3)
    st.warning("**Note :** For optimal results, input values between 6 - 8 bars or any number of bars.")

    coolant_temperature = st.text_input("Enter coolant temperature value : ", key = 4)
    st.warning("**Note :** For optimal results, input values between 85°C - 95°C or any number of degrees.")

    hydraulic_oil_temperature = st.text_input("Enter hydraulic oil temperature value : ", key = 5)
    st.warning("**Note :** For optimal results, input values between 40°C - 60°C or any number of degrees.")

    spindle_bearing_temperature = st.text_input("Enter hydraulic oil temperature value : ", key = 6)
    st.warning("**Note :** For optimal results, input values between 60°C - 80°C or any number of degrees.")

    spindle_vibration = st.text_input("Enter spindle vibration value : ", key = 7)
    st.warning("**Note :** For optimal results, input values below 1 µm (mm/s) or any number of mm/s.")

    tool_vibration = st.text_input("Enter tool vibration value : ", key = 8)
    st.warning("**Note :** For optimal results, input values below 2.5 µm (mm/s) or any number of mm/s.")

    spindle_speed = st.text_input("Enter spindle_speed value : ", key = 9)
    st.warning("**Note :** For optimal results, input values between 500 - 3000 RPM or any number of RPM.")

    voltage = st.text_input("Enter voltage value : ", key = 10)
    st.warning("**Note :** For optimal results, input values between 12 to 14 volts or any number of volts.")

    torque = st.text_input("Enter Torque value : ", key = 11)
    st.warning("**Note :** For optimal results, input values between 20 to 30 Nm or any number of Nm values.")

    cutting = st.text_input("Enter cutting value : ", key = 12)
    st.warning("**Note :** For optimal results, input values between 5 to 10 kN or any number of kN values.")

    machine_id = st.selectbox("select machine_id : " , df["Machine_ID"].unique().tolist())

    but = st.button("Click here to predict",use_container_width=True,type="primary")
if but:
    sample_test_case = np.array([[
                Month,  # Month
                np.log1p(float(hydraulic_pressure)),  # Hydraulic_Pressure_log
                np.log1p(float(coolant_pressure)),  # Coolant_Pressure_log
                np.log1p(float(air_pressure)),  # Air_System_Pressure_log
                np.log1p(int(coolant_temperature)),  # Coolant_Temperature_log
                np.sqrt(int(hydraulic_oil_temperature)),  # Hydraulic_Oil_Temperature_sqrt
                np.sqrt(abs(df['Spindle_Bearing_Temperature(°C)'].max() - int(spindle_bearing_temperature))),  # Spindle_Bearing_Temperature_reflect_sqrt
                np.sqrt(abs(df['Spindle_Vibration(µm)'].max() - float(spindle_vibration))),  # Spindle_Vibration_reflect_sqrt
                float(tool_vibration),  # Tool_Vibration_normal
                np.log1p(float(spindle_speed)),  # Spindle_Speed_log
                np.sqrt(abs(df['Voltage(volts)'].max() - float(voltage))),  # Voltage_reflect_sqrt
                np.log1p(float(torque)),  # Torque_log
                np.log1p(float(cutting)),  # Cutting_log
                machine_id  # Machine_id
            ]])

    # Reshape the Machine_id feature to a 2D array
    Machine_id_feature = sample_test_case[:,-1].reshape(-1, 1)

    # Apply OneHotEncoder
    Machine_ohe = OH.transform(Machine_id_feature).toarray()

    # Concatenate the one-hot encoded Machine_id with the rest of the sample test case
    sample_test = np.concatenate((sample_test_case[:,:-1] , Machine_ohe), axis=1)

    scaled_sample_test = ss.transform(sample_test)

    y_pred = model.predict(scaled_sample_test)[0]

    predict_prob = model.predict_proba(scaled_sample_test)[0]

    max_prob = predict_prob[np.argmax(predict_prob)] * 100

    




    # Adding custom CSS styles using HTML within Markdown
    st.markdown(
        """
        <style>
            .grid-container {
                display: grid; /* Setting display property to grid for grid layout */
                grid-template-columns: repeat(3, 1fr); /* Setting grid to have 3 columns */
                grid-gap: 20px; /* Adding gap between grid items */
            }
            .grid-item {
                background-color: #f9f9f9; /* Setting background color for grid items */
                padding: 20px; /* Adding padding around grid items */
                border-radius: 5px; /* Adding border radius to grid items */
                margin-bottom: 20px; /* Adding margin to create space between rows */
            }
            .grid-item h2 {
                color: #333333; /* Setting color for h2 headings */
                margin-bottom: 10px; /* Adding margin below h2 headings */
            }
            .grid-item h3 {
                color: #556b2f;  /* Setting color for h3 headings */
                margin-bottom: 10px; /* Adding margin below h3 headings */
                font-size: 40px;
            }
            .grid-item p {
                color: black; /* Setting color for paragraph text */
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    if y_pred == 1:
        # Creating HTML elements with extracted data and adding them to the columns list
        st.markdown(f"<div class='grid-item'><h2 style='color: #581845;'>🛠️ Machine downtime predicted outcome:<h3>{int(max_prob)}% chance of Machine gets failure</h3></h2></div>",unsafe_allow_html=True)
    else:
        # Creating HTML elements with extracted data and adding them to the columns list
        st.markdown(f"<div class='grid-item'><h2 style='color: #581845;'>🛠️ Machine downtime predicted outcome:<h3>{int(max_prob)}% chance of No machine Failure</h3></h2></div>",unsafe_allow_html=True)
    
    # Create a DataFrame with the probabilities
    df_probabilities = pd.DataFrame({
        'Class': ['No_Machine_Failure', 'Machine_Failure'],
        'Probability': predict_prob * 100
    })

    prob = []
    for i in df_probabilities["Probability"]:
        prob.append(str(int(np.round(i))) + "%")
    
    df_probabilities["Probability_percentage"] = prob
    # Plot the bar plot using Plotly Express
    fig = px.bar(
        df_probabilities, 
        x='Class', 
        y='Probability', 
        title='Predicted Probabilities for Machine Failure',
        labels={'Class': 'Class label 📊 ', 'Probability': 'Predicted Probability 🎲 '},
        color='Class',  # Use the 'Class' column to specify the bar colors
        color_discrete_map={'No_Machine_Failure': 'green', 'Machine_Failure': 'red'}  ,# Custom color mapping
        hover_data = "Probability_percentage"
    )


    st.plotly_chart(fig)

    
else:
    st.error("Please enter data to precit the machine downtime")