import dataretrieval.nwis as nwis
import pandas as pd

def get_streamflow(site_id):
    """
    Fetches real-time streamflow data for a given site from the USGS API.

    :param site_id: USGS site ID (e.g., "09380000" for Colorado River at Lees Ferry)
    :return: A formatted string with the latest streamflow data.
    """
    # Request real-time streamflow data
    df, meta = nwis.get_iv(sites=site_id, parameterCd="00060", period="1D")
    
    if df.empty:
        return "No streamflow data available for this site."

    # Extract latest streamflow reading
    latest_time = df.index[-1]
    latest_value = df["00060"].iloc[-1]  # '00060' is the parameter code for streamflow

    return f"The latest streamflow at site {site_id} is {latest_value} cubic feet per second (cfs) as of {latest_time}."
