import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(
    page_title="Case 3 - Laadpalen & Elektrisch vervoer",
    layout="wide"
)


@st.cache_data
def read_pkl():
    return pd.read_pickle('cars2.pkl')


cars_df = read_pkl()
cars_df['massa_rijklaar'] = cars_df['massa_rijklaar'].astype(int)


def bepaal_brandstof_v2(mass: int):
    if mass >= 1900:
        return 'elektrisch'
    elif 1700 <= mass < 1900:
        return 'hybride'
    else:
        return 'benzine'


cars_df['brandstof_omschrijving'] = cars_df['massa_rijklaar'].apply(bepaal_brandstof_v2)
cars_df['datum_tenaamstelling_dt'] = pd.to_datetime(cars_df['datum_tenaamstelling_dt'], errors='coerce')
cars_df['maand_jaar'] = cars_df['datum_tenaamstelling_dt'].dt.to_period('M')
grouped = cars_df.groupby(['maand_jaar', 'brandstof_omschrijving']).size().reset_index(name='Aantal_voertuigen')
grouped['Cumulatief_aantal_voertuigen'] = grouped.groupby('brandstof_omschrijving')['Aantal_voertuigen'].cumsum()
grouped['maand_jaar'] = grouped['maand_jaar'].astype(str)


class LinearRegressionModel:
    def __init__(self, dataframe, x_axis, y_axis):
        self.dataframe = dataframe
        self.x_axis = x_axis
        self.y_axis = y_axis

        self.x = None
        self.fit = None
        self.fit_function = None

    def model_fit(self):
        self.x = np.arange(self.dataframe[self.x_axis].size)
        self.fit = np.polyfit(self.x, self.dataframe[self.y_axis], deg=1)
        return self.fit

    def model_function(self):
        self.fit_function = np.poly1d(self.fit)
        return self.fit_function(self.x)

    def model_prediction(self, n: int):
        self.fit_function(self.dataframe[self.x_axis].size + n)
        return self.fit_function(self.x)

    def model_prediction_range(self, n: int):
        self.fit_function(self.dataframe[self.x_axis].size + np.arange(n))
        return self.fit_function(self.x)

    def model_parameters(self):
        if self.fit[1] < 0:
            self.fit[1] = 0
        return self.fit[0], self.fit[1]

    def calculate_r_squared(self):
        return r2_score(self.dataframe[self.y_axis], self.model_function())

    def calculate_rmse(self):
        return np.sqrt(mean_squared_error(self.dataframe[self.y_axis], self.model_function()))

    def calculate_mae(self):
        return mean_absolute_error(self.dataframe[self.y_axis], self.model_function())


df = grouped[grouped['brandstof_omschrijving'] == 'benzine']
amount_vehicles_model: LinearRegressionModel = LinearRegressionModel(df, 'maand_jaar', 'Cumulatief_aantal_voertuigen')
amount_vehicles_model.model_fit()
r_squared1 = amount_vehicles_model.calculate_r_squared()
rmse1 = amount_vehicles_model.calculate_rmse()
mae1 = amount_vehicles_model.calculate_mae()
amount_vehicles_model.model_function()
prediction = amount_vehicles_model.model_prediction_range(8)

df = grouped[grouped['brandstof_omschrijving'] == 'elektrisch']
amount_vehicles_model: LinearRegressionModel = LinearRegressionModel(df, 'maand_jaar', 'Cumulatief_aantal_voertuigen')
amount_vehicles_model.model_fit()
r_squared2 = amount_vehicles_model.calculate_r_squared()
rmse2 = amount_vehicles_model.calculate_rmse()
mae2 = amount_vehicles_model.calculate_mae()
amount_vehicles_model.model_function()
prediction2 = amount_vehicles_model.model_prediction_range(8)

df = grouped[grouped['brandstof_omschrijving'] == 'hybride']
amount_vehicles_model: LinearRegressionModel = LinearRegressionModel(df, 'maand_jaar', 'Cumulatief_aantal_voertuigen')
amount_vehicles_model.model_fit()
r_squared3 = amount_vehicles_model.calculate_r_squared()
rmse3 = amount_vehicles_model.calculate_rmse()
mae3 = amount_vehicles_model.calculate_mae()
amount_vehicles_model.model_function()
prediction3 = amount_vehicles_model.model_prediction_range(8)

st.markdown("## Prediction models", unsafe_allow_html=True)


@st.cache_resource
def prediction_figure():
    split_point = -3  # Adjust this to the point where predictions start

    fig = go.Figure()

    # Benzine (actual - solid line)
    fig.add_trace(go.Scatter(
        x=df['maand_jaar'][:split_point+1],
        y=prediction[:split_point+1],
        mode='lines',
        name='Benzine - Actual',
        line=dict(color='aqua', dash='solid')  # Solid line for actual data
    ))

    # Benzine (predicted - dashed line)
    fig.add_trace(go.Scatter(
        x=df['maand_jaar'][split_point:],
        y=prediction[split_point:],
        mode='lines',
        name='Benzine - Predicted',
        line=dict(color='aqua', dash='dash')  # Dashed line for predicted data
    ))

    # Elektrisch (actual - solid line)
    fig.add_trace(go.Scatter(
        x=df['maand_jaar'][:split_point+1],
        y=prediction2[:split_point+1],
        mode='lines',
        name='Elektrisch - Actual',
        line=dict(color='red', dash='solid')  # Solid line for actual data
    ))

    # Elektrisch (predicted - dashed line)
    fig.add_trace(go.Scatter(
        x=df['maand_jaar'][split_point:],
        y=prediction2[split_point:],
        mode='lines',
        name='Elektrisch - Predicted',
        line=dict(color='red', dash='dash')  # Dashed line for predicted data
    ))

    # Hybride (actual - solid line)
    fig.add_trace(go.Scatter(
        x=df['maand_jaar'][:split_point+1],
        y=prediction3[:split_point+1],
        mode='lines',
        name='Hybride - Actual',
        line=dict(color='purple', dash='solid')  # Solid line for actual data
    ))

    # Hybride (predicted - dashed line)
    fig.add_trace(go.Scatter(
        x=df['maand_jaar'][split_point:],
        y=prediction3[split_point:],
        mode='lines',
        name='Hybride - Predicted',
        line=dict(color='purple', dash='dash')  # Dashed line for predicted data
    ))

    # Layout configuration
    fig.update_layout(
        title="Linear regression prediction model of the amount of vehicles",
        xaxis_title="Date",
        yaxis_title="Amount of cumulative vehicles",
    )

    fig.update_layout(yaxis_range=[0, 1.1 * max(prediction2)])

    return st.plotly_chart(fig)


st.markdown(
    '''
    With the current energy transition, there is a boom in the amount of electric cars being sold.
    But how many electric cars will there be in the foreseeable future?
    With Linear Regression Model (LR Models), this can be predicted based on the current data.
    For these LR Models, the Numpy library is used.
    With Numpy, a polyfit with Least Squares Fitting is being used.
    Polyfit meaning, finding the best scalar values for the constants in a given function.
    Given the function $f(x) = ax + b$, the best scalar values will be found for the constants $a$ and $b$.  
    
    **[Medium - Simple linear regression fit and prediction on time series data with visualization in python](https://ishan-mehta17.medium.com/simple-linear-regression-fit-and-prediction-on-time-series-data-with-visualization-in-python-41a77baf104c)**
    '''
)

fig = prediction_figure()




st.markdown("## Errors Visualization", unsafe_allow_html=True)

st.markdown(
    '''
    Given the above Linear Regression models, what is the accuracy of these models.
    One of the most common ways to test the accuracy is to calculate the $r^2$ (r-squared) value.
    This value indicates the fitting of the model, with 1 making it a perfect fit and 0 making it the worst possible fit. 
    The Mean Absolute Error (MAE) indicates the average absolute distance when making a prediction. 
    A low MAE means that the models is correctly predicting.
    At last, the Root Mean Squared Error (RMSE) gives the square root of the average squared distance between actual and predicted value.

    **[FreeCodeCamp - Top Evaluation Metrics for Regression Problems in Machine Learning](https://www.freecodecamp.org/news/evaluation-metrics-for-regression-problems-machine-learning/)**
    '''
)

fuel_type = st.radio('Select fuel type', ('benzine', 'elektrisch', 'hybride'))

df = grouped[grouped['brandstof_omschrijving'] == fuel_type]


def fuel_type_prediction(fuel_types):
    if fuel_types == 'benzine':
        return prediction
    elif fuel_types == 'elektrisch':
        return prediction2
    else:
        return prediction3


predicted_values = fuel_type_prediction(fuel_type)

errors = df['Cumulatief_aantal_voertuigen'] - predicted_values


def error_fig():
    fig = go.Figure()

    # Add actual data points as scatter markers
    fig.add_trace(go.Scatter(
        x=df['maand_jaar'],
        y=df['Cumulatief_aantal_voertuigen'],
        mode='markers',
        marker_color='blue',
        name='Actual data'
    ))

    # Add prediction line
    fig.add_trace(go.Scatter(
        x=df['maand_jaar'],
        y=predicted_values,
        mode='lines',
        name='Prediction model',
        line=dict(color='red', dash='solid')
    ))

    # Add error lines from actual data points to predicted values
    for i in range(len(df)):
        fig.add_trace(go.Scatter(
            x=[df['maand_jaar'].iloc[i], df['maand_jaar'].iloc[i]],  # x positions (same for both ends)
            y=[df['Cumulatief_aantal_voertuigen'].iloc[i], predicted_values[i]],  # y positions (actual and predicted)
            mode='lines',
            line=dict(color='gray', dash='dash'),  # Style for error lines
            showlegend=False  # Hide legend for these lines
        ))

    fig.update_layout(
        title="Actual data versus fitted model",
        xaxis_title="Date",
        yaxis_title="Amount of cumulative vehicles",
    )

    return st.plotly_chart(fig)


fig2 = error_fig()

stats_date = {
    'Fuel type': ['Gasoline', 'Electric', 'Hybrid'],
    'r-squared': [r_squared1, r_squared2, r_squared3],
    'MAE': [round(mae1), round(mae2), round(mae3)],
    'RMSE': [round(rmse1), round(rmse2), round(rmse3)]
}

stats_dataframe = pd.DataFrame(stats_date)
stats_dataframe.set_index('Fuel type', inplace=True)

st.write(
    '''
    At the given DataFrame below, the $r^2$, MAE, and RMSE value is calculated for each model of the different fuel types.   
    '''
)

st.dataframe(data=stats_dataframe, width=5000)





@st.cache_data
def rwd_api():
    return pd.read_json("https://opendata.rdw.nl/resource/m9d7-ebf2.json?$limit=100000")


RDWvoertuigen_df = rwd_api()

# Omzetten van datum naar datetime-formaat
RDWvoertuigen_df['datum_eerste_tenaamstelling_in_nederland'] = pd.to_datetime(
    RDWvoertuigen_df['datum_eerste_tenaamstelling_in_nederland'], format='%Y%m%d')
# set 'date' column to proper format, filter a new column holding the year and numeric month.
RDWvoertuigen_df['year'] = RDWvoertuigen_df['datum_eerste_tenaamstelling_in_nederland'].dt.year
# print(RDWvoertuigen_df['year'].value_counts())

non_cars = ['Middenasaanhangwagen', 'Motorfiets', 'Aanhangwagen', 'Bromfiets', 'Oplegger', 'Driewielig motorrijtuig',
            'Autonome aanhangwagen', 'Land- of bosb aanhw of getr uitr stuk', 'Mobiele machine',
            'Motorfiets met zijspan']
RDWvoertuigen_df.drop(RDWvoertuigen_df.loc[RDWvoertuigen_df['voertuigsoort'].isin(non_cars)].index, inplace=True)
RDWvoertuigen_df['aantal_cilinders'] = RDWvoertuigen_df['aantal_cilinders'].fillna(0)

elektrische_merken = RDWvoertuigen_df[RDWvoertuigen_df['aantal_cilinders'] == 0]

# Streamlit page title
# st.title("EV popularity via registration count from RDW datasets")

st.markdown("## EV popularity via registration", unsafe_allow_html=True)


# Selectbox for choosing plot type
plot_type = st.selectbox("Kies een plot type:", ["20 meest geregistreerde merken", "20 meest geregistreerde modellen"])

# plots data for brand choice
if plot_type == "20 meest geregistreerde merken":
    # set up variable for most popular brands from RDW and the average prices for each these brands.
    populaire_merken = elektrische_merken['merk'].value_counts().head(20)
    average_prices = elektrische_merken.groupby('merk')['catalogusprijs'].mean().loc[populaire_merken.index]

    # Plots bars for brand count
    fig, ax1 = plt.subplots(figsize=(10, 5))
    populaire_merken.plot(kind='bar', color='skyblue', ax=ax1, position=0, width=0.4)
    ax1.set_title('20 meest populaire elektrische merken')
    ax1.set_xlabel('Merk', fontsize=10)
    ax1.set_ylabel('Aantal registraties', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.grid(axis='y')

    # Plots bars for brand average price
    ax2 = ax1.twinx()
    average_prices.plot(kind='bar', color='orange', ax=ax2, position=1, width=0.4)
    ax2.set_ylabel('Gemiddelde catalogusprijs', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    st.pyplot(fig)

# plots data for model choice
elif plot_type == "20 meest geregistreerde modellen":
    # set up variable for most popular models from RDW and the average prices for each these models.
    populaire_modellen = elektrische_merken['handelsbenaming'].value_counts().head(20)
    average_model_prices = elektrische_merken.groupby('handelsbenaming')['catalogusprijs'].mean().loc[
        populaire_modellen.index]

    # Plots bars for model count
    fig, ax1 = plt.subplots()
    populaire_modellen.plot(kind='bar', color='lightgreen', ax=ax1, position=0, width=0.4)
    ax1.set_title('Populaire modellen elektrische voertuigen')
    ax1.set_xlabel('Model', fontsize=10)
    ax1.set_ylabel('Aantal voertuigen', color='lightgreen')
    ax1.tick_params(axis='y', labelcolor='lightgreen')
    ax1.grid(axis='y')

    # Plots bars for model average price
    ax2 = ax1.twinx()
    average_model_prices.plot(kind='bar', color='red', ax=ax2, position=1, width=0.4)
    ax2.set_ylabel('Gemiddelde catalogusprijs', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    st.pyplot(fig)
