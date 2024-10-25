import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(
    page_title="Laadpalen - versie 2",
    layout="wide"
)

st.sidebar.markdown(
    '''
    # Sections  
    - [Charging station per province](#charging-station-per-province)  
    - [Berekenen oplaadduur](#berekenen-oplaadduur)  
    - [Bezet houden laadpaal in uren](#bezet-houden-laadpaal-in-uren)  
    - [Favoriete laadmomenten van de gebruiker](#favoriete-laadmomenten-van-de-gebruiker)  
    - [Gemiddelde prijs voor een kWh in Euro](#gemiddelde-prijs-voor-een-kwh-in-Euro)  
    - [Stroomkosten laadpaal](#stroomkosten-laadpaal)
    ''',
    unsafe_allow_html=True
)

st.header('Charging station per province')

@st.cache_data
def API_shivano():
    url = "https://api.openchargemap.io/v3/poi"
    api_key = "a887fc1e-bb00-417f-9dc2-be020b34d5d1"
    country_code = "NL"  # Landcode voor Nederland
    max_results = 7957  # Maximum aantal resultaten per aanroep

    return requests.get(url, params={
        'key': api_key,
        'countrycode': country_code,
        'maxresults': max_results
    })


# Maak een DataFrame van de laadpunten
df = pd.DataFrame(API_shivano().json())

# Groepeer op provincie en tel het aantal laadpunten
province_counts = df['AddressInfo'].apply(lambda x: x.get('StateOrProvince')).value_counts()

# Combineer verschillende namen voor dezelfde provincies
combined_provinces = {
    'Utrecht': ['Utrecht', 'UT', 'UTRECHT'],
    'Gelderland': ['Gelderland'],
    'Noord-Holland': ['North Holland', 'Noord-Holland', 'North-Holland', 'Noord Holand'],
    'Zuid-Holland': ['Zuid-Holland', 'Zuid Holland', 'South Holland', 'ZH'],
    'Zeeland': ['Zeeland', 'Seeland'],
    'Noord-Brabant': ['Noord-Brabant', 'North Brabant', 'Noord Brabant'],
    'Overijssel': ['Overijssel'],
    'Flevoland': ['Flevoland'],
    'Limburg': ['Limburg'],
    'Groningen': ['Groningen'],
    'Drenthe': ['Drenthe'],
    'Friesland': ['Friesland', 'Fryslân', 'FRL']
}

# Tel de laadpunten per provincie
final_counts = {}
for province, names in combined_provinces.items():
    final_counts[province] = sum(province_counts.get(name, 0) for name in names)

# Maak een DataFrame voor de final_counts
final_df = pd.DataFrame(list(final_counts.items()), columns=['Province', 'Charging Points'])
final_df = final_df.sort_values(by='Charging Points', ascending=False)

# Plot de gegevens met Plotly in een enkele groene kleur
fig = px.bar(final_df, x='Province', y='Charging Points',
             title='Verdeling van Laadpunten per Provincie',
             labels={'Province': 'Provincie', 'Charging Points': 'Aantal Laadpunten'},
             color_discrete_sequence=['green']
             )

fig.update_layout(xaxis_title='Provincie', yaxis_title='Aantal Laadpunten', xaxis_tickangle=-45)
st.plotly_chart(fig)


st.header('Berekenen oplaadduur')

st.write('### Invullen gegevens')
col1, col2 = st.columns(2)
Accu_cappaciteit_kWh = col1.number_input('Accu capaciteit auto in kW', min_value=0, value=200)
Benodige_percentage = col1.number_input('Benodigde Percentage', min_value=0, max_value=100, value=10)
Cappaciteit_laadpaal = col1.number_input('Capaciteit laadpaal in kW', min_value=0, value=20)


#Calculaties
Batterij = Benodige_percentage / 100 
benodigde_kwh = Batterij * Accu_cappaciteit_kWh


Oplaadtijd = benodigde_kwh / Cappaciteit_laadpaal
st.metric(label='### Oplaadtijd in uren', value=Oplaadtijd)



st.header('Bezet houden laadpaal in uren')


@st.cache_data
def compleetdataframe():
    return pd.read_csv('pages/compleetdataframe.csv')


laadpaal = compleetdataframe()
fig = px.scatter(laadpaal, x='month_start', y='overtime', title='Scatterplot van Overtime', color_discrete_sequence=['green'])
st.plotly_chart(fig)




# Create a mapping for month numbers to month names
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# Apply the mapping to the 'month' column
laadpaal['month_start'] = laadpaal['month_start'].map(month_names)

# Group by 'month_start' and sum 'overtime'
monthly_overtime = laadpaal.groupby('month_start')['overtime'].sum().reset_index()

# Create a function to filter the data based on the checkbox
def apply_filter(data):
    if st.checkbox('Filter absurd values'):
        return data[data['overtime'] < 350]
    return data

# Apply the filter to the data
filtered_data = apply_filter(laadpaal)

# Create a bar chart using Plotly Express with the filtered data
fig2 = px.bar(filtered_data, x='month_start', y='overtime', labels={'overtime': 'Sum of occupied'}, title='Sum of occupied by Month', color_discrete_sequence=['green'])

# Show the bar chart
st.plotly_chart(fig2)




st.header('Favoriete laadmomenten van de gebruiker')


@st.cache_data
def laadpaal_data():
    return pd.read_csv("pages/laadpaaldata2.csv")


# Load the data from the CSV file
laadpaal = laadpaal_data()

# Create a dropdown menu for selecting the hour column
hour_col = st.selectbox("Select Hour Column", ['Starten laden', 'Stoppen laden'], index=0)

if hour_col == 'Starten laden':
    hourly_counts = laadpaal.groupby('hour_start').size()
    x_label = "Starten laden"
else:
    hourly_counts = laadpaal.groupby('hour_end').size()
    x_label = "Stoppen laden"

# Create a bar chart with Plotly
fig2 = go.Figure(data=[
    go.Bar(x=hourly_counts.index, y=hourly_counts.values, marker_color='green')
])

# Set title and labels
fig2.update_layout(
    title="Verdeling van starten & stoppen laden per uur",
    xaxis_title=x_label,
    yaxis_title="Start/stop momenten",
    width=1250,  # Increase width (default width is ~500, so 2.5x)
    height=800   # Increase height (default height is ~400, so 2x)
)

# Show the figure
st.plotly_chart(fig2)






st.header('Gemiddelde prijs voor een kWh in Euro')

# Load your dataset


@st.cache_data
def random_dataset():
    return pd.read_csv('pages/2013tm2023.csv')


df_jaren = random_dataset()

# Create a mapping for month numbers to month names
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# Apply the mapping to the 'month' column
df_jaren['month'] = df_jaren['month'].map(month_names)

# Create a function to generate the histogram based on the selected x-axis
def create_histogram(x_axis_column):
    fig = go.Figure()
    
    # Add histogram trace
    fig.add_trace(go.Histogram(
        x=df_jaren[x_axis_column], # x-axis: dynamically based on user selection
        y=df_jaren['Prijs'],  # y-axis: Prijs
        histfunc='avg',    # average of 'Prijs' values
        marker_color='green',  # set color of bars
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Histogram of price by {x_axis_column.capitalize()}',
        xaxis_title=x_axis_column.capitalize(),
        yaxis_title='Average price for a kWh in Euro',
        xaxis=dict(tickangle=45 if x_axis_column == 'month' else 0), # Rotate if it's month
        bargap=0.2,
        width=1250, # Increase width (default width is ~500, so 2.5x)
        height=800  # Increase height (default height is ~400, so 2x)
    )
    
    return fig

# Add dropdown menu for selecting x-axis
x_axis_option = st.selectbox('Select x-axis:', ['Yearly', 'Monthly', 'Hourly'])

if x_axis_option == 'Yearly':
    x_axis_column = 'year'
elif x_axis_option == 'Monthly':
    x_axis_column = 'month'
else:
    x_axis_column = 'hour'

# Create the plot based on selected x-axis
fig3 = create_histogram(x_axis_column)

# Display the plot
st.plotly_chart(fig3)

st.markdown("<p style='text-align: center;'>https://energie.anwb.nl/actuele-tarieven</p>", unsafe_allow_html=True)
    
    
    


st.header('Stroomkosten laadpaal')
# Load the data
merged_df = compleetdataframe()

# Create a mapping for month numbers to month names
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# Apply the mapping to the 'month' column
merged_df['month_start'] = merged_df['month_start'].map(month_names)

# Dropdown menu for selecting 'Total' or 'Months'
plot_type = st.selectbox('Select Plot Type:', ['Total', 'Months'])

# Checkbox to include/exclude BTW in charge costs
btw_included = st.checkbox('BTW Included')

# Adjust column based on BTW inclusion
costs_column = 'Charge Costs BTW included' if btw_included else 'Charge Costs BTW excluded'

# Create the plot based on the selected plot type
if plot_type == 'Total':
    # Group by year and sum the costs
    total_costs = merged_df.groupby('year')[costs_column].sum().reset_index()
    fig = px.bar(total_costs, x='year', y=costs_column, title='Total Charge Costs per Year', color_discrete_sequence=['green'])
else:
    fig = px.histogram(merged_df, x='month_start', y=costs_column, histfunc='sum', title='Histogram of Charge Costs per Month', color_discrete_sequence=['green'])

fig.update_layout(xaxis_title='Month', yaxis_title='Charge Costs', bargap=0.2, width=1250, 
height=800)   

# Display the selected plot in Streamlit
st.plotly_chart(fig)

st.markdown("<p style='text-align: center;'>https://energie.anwb.nl/actuele-tarieven</p>", unsafe_allow_html=True)
