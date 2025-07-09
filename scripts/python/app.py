
# import streamlit library
import streamlit as st
# import numpy and pandas
import pandas as pd
import numpy as np
# import ploty Library
import plotly.express as px

# Generate Random_Data for the model to use for prediction
# You also use CSV file data which is downloaded from Static site
# and even you can take it from API (Application Programming Interface)
def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

# import scikit-learn packages 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression

# Training model function
def train_model():
    df = generate_house_data(n_samples=100)
    x = df[['size']]
    y = df[['price']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LinearRegression()
    model.fit(x_train.values, y_train.values)

    return model

# Define main() function 
def main():
    # the title of the streamlit app
    st.title("Simple Linear Regression House Prediction Model")
    
    # Call write function of streamlit 
    st.write("Put in your House Size to know its price")

    # Calling the train_model Function
    model = train_model()

    # Asking function about the size of the House area
    size = st.number_input('House Area (foot-Square)', min_value = 500, max_value = 2000, value = 1500)

    # Create a button 
    if st.button('Predict Price'):
        predicted_price = model.predict([[size]])
        st.success(f'Estimated Price: ${predicted_price[0][0]:.2f}')

        # Calls house_data form
        df = generate_house_data()

        # The Scatter plot of the x and y values.
        fig = px.scatter(df, x='size', y='price', title = 'Depiction of House & Price')
        fig.add_scatter(x = [size], y = [predicted_price[0][0]],
                    mode = 'markers',
                    marker = dict(size = 15, color = 'red'), name = 'Prediction')
        # Depicting the chart
        st.plotly_chart(fig)

# Calls the main function
if __name__ == '__main__':
    main()













