import dataretrieval.nwis as nwis
import pandas as pd

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

# Example Usage
if __name__ == "__main__":
    site_id = "01589000"  # Example USGS site
    result = get_streamflow_data(site_id)
    print(result)
