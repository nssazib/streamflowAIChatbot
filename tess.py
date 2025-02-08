
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hyswap
import google.generativeai as genai
import dataretrieval.nwis as nwis
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hyswap
import google.generativeai as genai
import dataretrieval.nwis as nwis
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
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

print(format_site_id(1589000))
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




def get_streamflow_data(site_id, start_date="2022-01-01", end_date="2022-01-02"):
    """
    Fetches streamflow and water quality data for a given USGS site.

    :param site_id: USGS site ID (e.g., "03339000")
    :param start_date: Start date for data retrieval (YYYY-MM-DD)
    :param end_date: End date for data retrieval (YYYY-MM-DD)
    :return: A dictionary with streamflow, water quality, and site information.
    """
    try:
        # Get streamflow (instantaneous values)
        streamflow_df = nwis.get_record(sites=site_id, service='iv', start=start_date, end=end_date)

        # Get water quality samples
        #water_quality_df = nwis.get_record(sites=site_id, service='qwdata', start=start_date, end=end_date)

        # Get site information
        site_info_df = nwis.get_record(sites=site_id, service='site')

        # Extract meaningful site name (if available)
        site_name = site_info_df.loc[site_id, 'station_nm'] if site_id in site_info_df.index else "Unknown Site"

        # Prepare output
        data = {
            "site_name": site_name,
            "streamflow": streamflow_df.to_dict(),
           # "water_quality": water_quality_df.to_dict(),
            "site_info": site_info_df.to_dict()
        }

        return data

    except Exception as e:
        return {"error": str(e)}



def get_streamflow_data2(site_id, start_date, end_date):
    """Fetch daily streamflow data from NWIS for the given site."""
    flow_data, _ = nwis.get_dv(sites=site_id, parameterCd='00060', start=start_date, end=end_date)

    if flow_data.empty:
        return None

    # Replace preliminary, non-valid observations (-999999) with NaN
    flow_data['00060_Mean'] = flow_data['00060_Mean'].replace(-999999, np.nan)

    # Filter for approved data
    approved_flow_data = hyswap.utils.filter_approved_data(flow_data, '00060_Mean_cd')

    return approved_flow_data

def generate_hydrograph(site_id, data, plot_start="2021-10-01"):
    """Generates a hydrograph with a customized appearance."""
    if data is None or data.empty:
        return "No streamflow data available."

    fig, ax = plt.subplots(figsize=(12, 4))
    ax = hyswap.plot_hydrograph(
        data,
        data_column_name="00060_Mean",
        start_date=plot_start,
        title=f"Hydrograph for {site_id}",
        yscale='linear',
        ylab='Streamflow (cfs)',
        xlab='',
        color='#360919',
        ax=ax
    )

    # Grid Customization
    ax.grid(which="major", axis="y", lw=1.5)
    ax.grid(which="minor", axis="y", linestyle="dashed")
    ax.minorticks_on()

    # Save and show the plot
    hydrograph_filename = f"hydrograph_{site_id}.png"
    plt.savefig(hydrograph_filename)
    plt.show()

    return hydrograph_filename


def explain_graph(site_id, start_date, end_date, plots):
    plots=r'C:\Users\sazib\OneDrive\Nazmus_AI_engineer_plan\streamflow_proc\streamflowAIChatbot\hydrograph_01594440.png'
    """Uses LLM to generate a hydrologic interpretation of the graph."""
    prompt = f"""
    You are a hydrologist analyzing streamflow trends.
    Based on the hydrograph plot for USGS site {site_id} from {start_date} to {end_date}, interpret the data.
    
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

def generate_summary_stats(data):
    """Calculates summary statistics on approved streamflow data."""
    return hyswap.calculate_summary_statistics(data)
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
            st.text(nearest_gauge)
            if not nearest_gauge:
                st.error("No gauge found.")
                return
            site_id = str(int(nearest_gauge["nearest_gauge_id"])).zfill(8)
            #st.text(10)
            #site_id = str(site_id).strip()  # Convert to string and remove spaces
            st.text(site_id)
           # site_id=format_site_id(site_id)
            #st.text(site_id)
            st.success(f"Nearest Gauge: {nearest_gauge['nearest_gauge_name']} (Site ID: {site_id})")
       # with st.spinner("Fetching streamflow data..."):
            data = get_streamflow_data(site_id, start_date, end_date)
            if data is None:
                st.error("No streamflow data available.")
                return
            st.subheader("Hydrograph")
            #plot_name=generate_hydrograph(site_id, data)
            #st.image(plot_name)
      #  with st.spinner("Calculating summary statistics..."):
       #     stats = generate_summary_stats(data)
        #st.subheader("Summary Statistics")
        #st.write(stats)
        # with st.spinner("Generating AI Interpretation..."):
        #     explanation = explain_graph(site_id, start_date, end_date, plot_name)
        # st.subheader("AI Hydrologic Interpretation")
        # st.write(explanation)

if __name__ == "__main__":
    main()

