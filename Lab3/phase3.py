import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta

# --- Configuration ---
# NOTE: Replace with your actual Gemini API key, or load from environment variable
# Streamlit provides simple ways to handle secrets. For local testing, paste it here.
#THERE WAS AN API KEY HERE
MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MAX_RETRIES = 5

# --- Helper Functions for LLM Communication ---

def extract_sources(attributions):
    """Converts the raw grounding attribution array into a simplified array of objects."""
    if not attributions:
        return []
    sources = []
    for attr in attributions:
        if 'web' in attr and 'uri' in attr['web'] and 'title' in attr['web']:
            sources.append({'uri': attr['web']['uri'], 'title': attr['web']['title']})
    return sources

def fetch_with_exponential_backoff(payload, url, api_key, max_retries=MAX_RETRIES):
    """Implements exponential backoff for API retries using requests."""
    if not api_key:
        st.error("Gemini API_KEY is missing. Please provide your API key to run the prediction.")
        st.stop()

    full_url = f"{url}?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    for i in range(max_retries):
        try:
            response = requests.post(full_url, headers=headers, data=json.dumps(payload), timeout=30)
            
            if response.ok:
                return response.json()

            if response.status_code == 429 or response.status_code >= 500:
                st.warning(f"Attempt {i + 1} failed with status {response.status_code}. Retrying...")
                if i < max_retries - 1:
                    delay = (2 ** i) + (time.time() % 1)
                    time.sleep(delay)
                continue
            
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            st.warning(f"Request attempt {i + 1} failed: {e}. Retrying...")
            if i < max_retries - 1:
                delay = (2 ** i) + (time.time() % 1)
                time.sleep(delay)
            else:
                st.error(f"API call failed after maximum retries: {e}")
                st.stop()

    st.error("API call failed after maximum retries.")
    st.stop()


# --- Historical Data Fetching and Summarization ---

def fetch_and_summarize_historical_data(city, unitsurl):
    """
    Fetches 20 years of historical data using Open-Meteo and generates a summary for the LLM.
    Integrates the user's provided logic.
    """
    
    # Define 20-year historical window ending today
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=20*365)).strftime("%Y-%m-%d")
    
    st.info(f"Fetching 20 years of historical data for {city} from {start_date} to {end_date}...")
    
    try:
        # 1. Geocoding
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
        endpoint = requests.get(url)
        data = endpoint.json()
        pophigh = 0
        lat = 0
        long = 0
        
        if "results" not in data or len(data["results"]) == 0:
            st.error("City not found. Check spelling or try with a different city.")
            return None, None

        # Select the most populated city match
        for c in data["results"]:
            if "population" in c:
                if c["population"] > pophigh:
                    pophigh = c["population"]
                    lat = c["latitude"]
                    long = c["longitude"]
        
        if lat == 0 and long == 0 and len(data["results"]) > 0:
             # Fallback to first result if population data is missing
            lat = data["results"][0]["latitude"]
            long = data["results"][0]["longitude"]


        # 2. Archive Data Fetching
        if unitsurl == "fahrenheit":
            url2 = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&temperature_unit={unitsurl}"
        else:
            url2 = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean"
            
        endpoint2 = requests.get(url2)
        data2 = endpoint2.json()

        if "daily" not in data2 or "temperature_2m_mean" not in data2["daily"]:
            st.error("No temperature data available in this time frame. Try again.")
            return None, None
            
        mean_temps = data2["daily"]["temperature_2m_mean"]
        dates = data2["daily"]["time"]
        
        # 3. Data Analysis for LLM Summary
        if not mean_temps:
             st.error("Historical data array is empty.")
             return None, None

        unit_symbol = "°F" if unitsurl == "fahrenheit" else "°C"
        
        avg_temp = sum(mean_temps) / len(mean_temps)
        min_temp = min(mean_temps)
        max_temp = max(mean_temps)
        
        # Simple trend analysis: Compare first 5 years vs last 5 years
        num_years_for_trend = 5 * 365
        if len(mean_temps) > 2 * num_years_for_trend:
            first_period_avg = sum(mean_temps[:num_years_for_trend]) / num_years_for_trend
            last_period_avg = sum(mean_temps[-num_years_for_trend:]) / num_years_for_trend
            trend = last_period_avg - first_period_avg
            trend_str = f"The average temperature has shown a change of **{trend:.2f} {unit_symbol}** between the first 5 years and the last 5 years of data. "
            if trend > 0:
                trend_str += "This indicates a clear warming trend over the period."
            elif trend < 0:
                trend_str += "This indicates a slight cooling trend over the period."
            else:
                trend_str += "The overall temperature trend is stable."
        else:
             trend_str = "Insufficient data points for a detailed 5-year trend comparison."


        historical_summary = f"""
        HISTORICAL DATA SUMMARY (20-Year Analysis for {city} in {unit_symbol}):
        
        * **Time Span:** {dates[0]} to {dates[-1]}.
        * **Overall Average Mean Temperature:** {avg_temp:.2f} {unit_symbol}.
        * **20-Year Extreme Low:** {min_temp:.2f} {unit_symbol}.
        * **20-Year Extreme High:** {max_temp:.2f} {unit_symbol}.
        * **Long-Term Trend:** {trend_str}
        
        You MUST use these specific numbers and trends to form the foundation of your conceptual prediction.
        """
        
        return historical_summary.strip(), f"{unit_symbol}"

    except Exception as excep:
        st.error(f"Something else went wrong during data fetching: {excep}")
        return None, None


# --- LLM Prediction Logic ---

def predict_weather(city, target_date, historical_summary, unit_symbol):
    """
    Generates the full prompt by combining historical data and the query, 
    then calls the Gemini API.
    """
    
    # 2. Construct the Final Prompt for the LLM
    full_prompt = (
        f"Analyze the following detailed Historical Data Summary for {city}, paying close attention "
        f"to the recorded average, extremes, and long-term trend. Use this analysis, "
        f"along with real-time global climate context (from Search grounding), to generate a prediction.\n\n"
        f"--- HISTORICAL DATA START ---\n"
        f"{historical_summary}\n"
        f"--- HISTORICAL DATA END ---\n\n"
        f"Prediction Query: Based on this historical data and current climate knowledge, "
        f"provide a detailed conceptual weather forecast for {city} on {target_date}. "
        f"The temperature prediction MUST be primarily in {unit_symbol}, but also include the secondary unit."
    )
    
    # Define the AI persona (System Prompt)
    system_prompt = """You are a conceptual, long-range atmospheric modeling AI (L-WPM-2.5). 
    Your task is to generate a plausible, detailed, and creative weather scenario for the future date requested by the user, explicitly analyzing the provided 'HISTORICAL DATA SUMMARY'.

    1.  **Analyze Premise:** Use the user's provided historical data summary as the primary input for determining multi-year climate tendencies.
    2.  **Grounding:** Use real-time data from Google Search to contextualize the current climate, seasonal expectations, and geography of the specified location.
    3.  **Output Format:** Your response must be a single, cohesive paragraph. It must include the predicted conditions (e.g., cloudy, severe storms, clear), a temperature range (in BOTH Fahrenheit and Celsius), and a summary of the wind patterns.
    4.  **Tone:** Be authoritative and provide a confident, detailed narrative of the conceptual future weather.
    """

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # Call LLM
    try:
        result = fetch_with_exponential_backoff(payload, MODEL_URL, API_KEY)
        
        candidate = result.get('candidates', [{}])[0]
        
        if candidate and candidate.get('content') and candidate['content'].get('parts') and candidate['content']['parts'][0].get('text'):
            
            text = candidate['content']['parts'][0]['text']
            sources = extract_sources(candidate.get('groundingMetadata', {}).get('groundingAttributions'))

            st.subheader("AI Prediction Scenario")
            st.markdown(f"**Target:** {city} on {target_date}")
            st.markdown("---")
            st.write(text)
            
            st.subheader("Grounding Sources (Real-time Context)")
            if sources:
                for source in sources:
                    st.markdown(f"**[{source['title']}]**({source['uri']})")
            else:
                st.write("No external web data was required for this conceptual prediction.")

        else:
            st.error("Could not generate a prediction. Check API response structure or token limits.")
            st.json(result) # Show raw result for debugging
            
    except Exception as e:
        st.exception(e)


# --- Streamlit Application Layout ---

def main():
    st.set_page_config(page_title="Gemini Long-Range Weather Predictor", layout="centered")
    st.title("🌌 Gemini-Powered Long-Range Weather Predictor")
    st.markdown("""
        This tool uses **20 years of real historical temperature data** (via Open-Meteo) 
        combined with the **Gemini LLM** to generate a *conceptual* forecast one year in the future.
        """)

    if not API_KEY:
        st.warning("⚠️ Please enter your Gemini API Key in the Python file (`API_KEY = ...`) to enable the prediction feature.")

    with st.sidebar:
        st.header("1. Data Parameters")
        city = st.text_input("Target City Name", "London")
        
        # Set target date to 1 year from now
        default_date = datetime.now() + timedelta(days=365)
        target_date = st.date_input("Target Date for Prediction", default_date)
        
        units = st.radio("Temperature Units", ["Celsius", "Fahrenheit"])
        unitsurl = "fahrenheit" if units == "Fahrenheit" else "celsius"

        st.header("2. Prediction")
        
        if st.button("Generate Conceptual Forecast 🚀", disabled=not API_KEY):
            if not city.strip():
                st.error("Please enter a valid city.")
                return

            with st.spinner(f"Step 1/2: Fetching 20 years of historical data for {city}..."):
                historical_summary, unit_symbol = fetch_and_summarize_historical_data(city, unitsurl)
            
            if historical_summary:
                st.session_state['summary'] = historical_summary
                st.session_state['city'] = city
                st.session_state['target_date'] = target_date.strftime("%Y-%m-%d")
                st.session_state['unit_symbol'] = unit_symbol
                st.session_state['show_prediction'] = True
            
            st.rerun()

    if 'show_prediction' in st.session_state and st.session_state['show_prediction']:
        st.subheader("Historical Data Analysis Sent to AI")
        st.code(st.session_state['summary'], language='markdown')
        
        with st.spinner("Step 2/2: Modeling atmospheric trends with Gemini..."):
            predict_weather(
                st.session_state['city'], 
                st.session_state['target_date'], 
                st.session_state['summary'],
                st.session_state['unit_symbol']
            )

if __name__ == "__main__":
    main()