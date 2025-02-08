import google.generativeai as genai
from streamflow import get_streamflow_data

# Configure Gemini API Key
genai.configure(api_key="AIzaSyAanrQp24DpsAQWPfVCBRfjP9NK8pya44k")  # Replace with your actual API key

# Initialize the Gemini model (using Gemini Flash for faster responses)
model = genai.GenerativeModel("gemini-2.0-flash")

def answer_basin_question(user_query, site_id, start_date="2022-01-01", end_date="2022-01-02"):
    """
    Uses Gemini to generate a natural language response to streamflow questions.

    :param user_query: User's question related to streamflow.
    :param site_id: USGS site ID.
    :return: AI-generated response.
    """
    data = get_streamflow_data(site_id, start_date, end_date)

    if "error" in data:
        return f"Error retrieving data: {data['error']}"

    # Construct prompt
    prompt = f"""
    You are a hydrology expert. A user asked: "{user_query}".
    Here is the retrieved data for {data['site_name']} (Site ID: {site_id}):

    - Streamflow Data: {data['streamflow']}

    - Site Information: {data['site_info']}

    Provide a clear, detailed, and accurate response based on this data.
    """

    # Generate response using Gemini
    response = model.generate_content(prompt)

    return response.text if response else "Sorry, I couldn't generate a response."

# Example Usage
if __name__ == "__main__":
    site_id = "03339000"
    question = "What is the streamflow and water quality at this location?"
    print(answer_basin_question(question, site_id))
