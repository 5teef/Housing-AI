import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Assume 'filtered_housing' is your DataFrame and 'split_data' is a function you've defined to split your data

# Split your data
feature_columns = ['median_income', 'ocean_proximity_encoded', 'longitude', 'latitude']
target_column = 'median_house_value'
X_train, X_val, X_test, y_train, y_val, y_test = split_data(filtered_housing, feature_columns, target_column)

# Train a Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Streamlit web app
def run():
    st.title('California Housing Price Prediction')

    # Input fields for the features
    median_income = st.number_input('Median Income')
    ocean_proximity_encoded = st.selectbox('Ocean Proximity', options=[0, 1, 2, 3])
    longitude = st.number_input('Longitude')
    latitude = st.number_input('Latitude')

    # Predict button
    if st.button('Predict'):
        input_data = [[median_income, ocean_proximity_encoded, longitude, latitude]]
        prediction = model.predict(input_data)
        st.success(f'Predicted House Value: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    run()
