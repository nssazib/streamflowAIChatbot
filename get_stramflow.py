from dataretrieval import nwis
import hyswap
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
import google.generativeai as genai

import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests
import dataretrieval.nwis as nwis
import pandas as pd
import os 
# Configure Gemini API Key
genai.configure(api_key="AIzaSyAanrQp24DpsAQWPfVCBRfjP9NK8pya44k")  # Replace with your actual API key

# Initialize the Gemini model (using Gemini Flash for faster responses)
model = genai.GenerativeModel("gemini-2.0-flash")
# Load USGS streamflow gauge locations from GeoJSON
geojson_file = r"C:\Users\sazib\OneDrive\Nazmus_AI_engineer_plan\streamflow_proc\streamflowAIChatbot\data\usgs_streamflow_gauges.geojson"
usgs_gdf = gpd.read_file(geojson_file)

def format_site_id(site_id):
    """
    Ensure the USGS site ID is an 8-digit string.
    - If 7 digits, prepend with '0'
    - If 6 digits, prepend with '00'
    - If already 8 digits, return as is.
    
    :param site_id: The site ID (int or string)
    :return: Properly formatted 8-digit site ID
    """
    site_id = str(site_id).strip()  # Convert to string and remove spaces

    # Ensure it has exactly 8 characters by adding leading zeros
    return site_id.zfill(8)


def get_lat_lon(city, state):
    """
    Convert city and state to latitude and longitude using Geopy.
    """
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.geocode(f"{city}, {state}")
    if location:
        return (location.latitude, location.longitude)
    return None

def find_nearest_gauge(city, state):
    """
    Find the nearest USGS streamflow gauge to a given city and state.
    """
    city_coords = get_lat_lon(city, state)
    if not city_coords:
        return "Invalid city/state provided."

    min_distance = float("inf")
    nearest_gauge = None

    for _, row in usgs_gdf.iterrows():
        gauge_coords = (row["latitude"], row["longitude"])
        distance = geodesic(city_coords, gauge_coords).kilometers

        if distance < min_distance:
            min_distance = distance
            nearest_gauge = row

    if nearest_gauge is not None:
        return {
            "nearest_gauge_id": nearest_gauge["site_no"],  # Update column name if needed
            "nearest_gauge_name": nearest_gauge["station_nm"],
            "latitude": nearest_gauge["latitude"],
            "longitude": nearest_gauge["longitude"],
            "distance_km": round(min_distance, 2)
        }
    else:
        return "No gauge found."



def get_streamflow_data(site_id, start_date, end_date):
    """Fetch daily streamflow data from NWIS for the given site."""
    
    flow_data, _ = nwis.get_dv(sites=site_id, parameterCd='00060', start=start_date, end=end_date)

    if flow_data.empty:
        return None

    # Replace preliminary, non-valid observations (-999999) with NaN
    flow_data['00060_Mean'] = flow_data['00060_Mean'].replace(-999999, np.nan)

    # Filter for approved data
    approved_flow_data = hyswap.utils.filter_approved_data(flow_data, '00060_Mean_cd')

    return approved_flow_data
import pandas as pd 
import pandas as pd  # Ensure pandas is imported
import matplotlib.pyplot as plt
import numpy as np
import hyswap
import pandas as pd
import matplotlib.pyplot as plt
import hyswap

def generate_hydrograph(site_id, data):
    """Generates a hydrograph for streamflow data."""
    fig, ax = plt.subplots(figsize=(12, 4))
    hyswap.plot_hydrograph(data, data_column_name="00060_Mean", start_date="2021-10-01", 
                            title=f"Hydrograph for {site_id}", yscale='linear', 
                            ylab='Streamflow (cfs)', xlab='', color='#360919', ax=ax)
    ax.grid(which="major", axis="y", lw=1.5)
    ax.grid(which="minor", axis="y", linestyle="dashed")
    ax.minorticks_on()
    save_path = os.path.join("C:\\Users\\sazib\\OneDrive\\Nazmus_AI_engineer_plan\\streamflow_proc\\streamflowAIChatbot\\data", f"hydrograph_{site_id}.png")
    plt.savefig(save_path)
    st.pyplot(fig)
    return save_path

def explain_graph(site_id, start_date, end_date, plots):
    plots=r'C:\Users\sazib\OneDrive\Nazmus_AI_engineer_plan\streamflow_proc\streamflowAIChatbot\hydrograph_01592500.png'
    """Uses LLM to generate a hydrologic interpretation of the graph."""
    prompt = f"""
    You are a hydrologist analyzing streamflow trends.
    Based on the hydrograph, flow duration curve, and cumulative flow plot for USGS site {site_id} from {start_date} to {end_date}, interpret the data.
    
    Consider:
    1. High-flow vs. low-flow trends.
    2. Seasonal patterns or anomalies.
    3. Any potential drought or flood conditions.
    
    The plots are:
    - Hydrograph: {plots}
   
    
    Provide a summary explaining these trends.
    """
    
    response = model.generate_content(prompt)
    
    return response.text


import streamlit as st
def main():
    """Streamlit app to fetch and analyze streamflow data."""
    st.title("Streamflow Analysis with AI")
    city = st.text_input("Enter City:")
    state = st.text_input("Enter State (Full Name):")
    start_date = st.date_input("Start Date:", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date:", pd.to_datetime("2023-12-31"))
    if st.button("Find Nearest Gauge & Analyze Streamflow"):
        with st.spinner("Finding nearest gauge..."):
            nearest_gauge = find_nearest_gauge(city, state)
            if not nearest_gauge:
                st.error("No gauge found.")
                return
           # site_id = str(nearest_gauge["site_no"]).zfill(8)
            site_id =format_site_id(int(nearest_gauge["nearest_gauge_id"]))
            gauge_name = nearest_gauge["nearest_gauge_name"]
            st.success(f"Nearest Gauge: {gauge_name } (Site ID: {site_id})")
        with st.spinner("Fetching streamflow data..."):
           # st.text(site_id)
           # data = get_streamflow_data(site_id, start_date, end_date)
            #StaID = '04288000'
            StaID =site_id
            flow_data = nwis.get_record(sites=StaID, parameterCd='00060', start='2020-01-01', service='dv')
            station_name = nwis.get_record(sites=StaID, service='site').loc[0, 'station_nm']

            #st.text(flow_data)
            if flow_data is None:
                st.error("No streamflow data available.")
            else:
                st.text("data feteched")
        #        return
        
        st.subheader("Hydrograph")
        st.text(site_id)
        plots_generated=generate_hydrograph(site_id,flow_data)
        st.text('hydrograph generated')
       # with st.spinner("Calculating summary statistics..."):
        #    stats = generate_summary_stats(data)
        #st.subheader("Summary Statistics")
        #st.write(stats)
        with st.spinner("Generating AI Interpretation..."):
            explanation = explain_graph(site_id, start_date, end_date,plots_generated)
        st.subheader("AI Hydrologic Interpretation")
        st.write(explanation)

if __name__ == "__main__":
    main()

