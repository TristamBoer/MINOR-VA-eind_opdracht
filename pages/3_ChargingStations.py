import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.title('Berekenen oplaadduur')

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



st.title('Bezet houden laadpaal in uren')


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




st.title('Favoriete laadmomenten van de gebruiker')


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






st.title('Gemiddelde prijs voor een kWh in Euro')

# Load your dataset


@st.cache_data
def random_dataset():
    return pd.read_csv('pages/2013tm2023.CSV')


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
    
    
    


st.title('Stroomkosten laadpaal')
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
