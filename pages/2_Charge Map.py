import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import requests
from folium.plugins import MarkerCluster

class Api:
    def __init__(self, url: str, limit: int = None):
        self.url = url
        self.limit = limit
        
        self.data = None
        self.response = None
        self.dataframe = None


    def get_data(self):
        if self.limit is not None:
            self.url = f'{self.url}?$limit={self.limit}'
        else:
            self.url = self.url
        
        self.response = requests.get(self.url)
        self.data = self.response.json()
        return self.data


    def get_df(self):
        if self.data is not None:
            self.dataframe = pd.DataFrame(self.data)
        return self.dataframe
    
api: Api = Api("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=7957&compact=true&verbose=false&key=93b912b5-9d70-4b1f-960b-fb80a4c9c017")

api_data = api.get_data()
api_df = api.get_df()
api_df = api_df.drop(columns=['GeneralComments', 'OperatorsReference', 'DataProvidersReference', 'MetadataValues', 'DateLastConfirmed'])
api_df['NumberOfPoints'] = api_df['NumberOfPoints'].ffill()

api_df2 = pd.json_normalize(api_df['AddressInfo'])

def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))

    legend_categories = ""
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"

    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map


def color_n_points(n: int):
    if n == 1:
        return 'black'
    elif n == 2:
        return 'gray'
    elif n == 3:
        return 'cadetblue'
    elif n == 4:
        return 'pink'
    elif n == 5:
        return 'red'
    elif n == 6:
        return 'orange'
    elif n == 8:
        return 'purple'
    elif n == 10:
        return 'green'
    elif n == 12:
        return 'blue'
    else:
        return 'beige'
    
    
def laadpaal_map_v2(df, df2):

    m = folium.Map(
        location=[52.0907374, 5.1214201],
        zoom_start=12,
        # scrollWheelZoom=False
    )
    
    marker_cluster = MarkerCluster().add_to(m)
    
    for i in df.index:
        n_points = df2.loc[i, 'NumberOfPoints']
        lat = df.loc[i, 'Latitude']
        lng = df.loc[i, 'Longitude']
        
        folium.Marker(
            location = (lat, lng),
            icon=folium.Icon(color=color_n_points(n_points)),
            tooltip='<b>Klik hier om de popup te zien</b>',
            popup=f'Aantal laadpunten: {n_points}'
        ).add_to(marker_cluster)
        
    m = add_categorical_legend(m, 'Aantal Laadpunten',
                           colors=['black', 'gray', 'cadetblue', 'pink', 'red', 'orange', 'purple', 'green', 'blue',
                                   'beige'],
                           labels=['1', '2', '3', '4', '5', '6', '8', '10', '12', '24'])
    
    return m


laadpaalmap = laadpaal_map_v2(api_df2, api_df)

with st.form(key="smth", border=False):
    st_folium(laadpaalmap, width=2000)
    st.write('Do not press the submit button! Page will reload and it changes nothing.')
    submitted = st.form_submit_button("Submit")
