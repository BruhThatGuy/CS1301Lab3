import streamlit as st
import requests
from datetime import datetime, timedelta

# ============================================================================
# SETUP
# ============================================================================
st.set_page_config(page_title="Weather Predictor", page_icon="🌤️")
st.title("🌤️ Long-Range Weather Predictor")
st.write("Predict weather up to 1 year in the future using historical data and AI!")

# Get API key from secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Please add GEMINI_API_KEY to your secrets!")
    st.stop()

# ============================================================================
# USER INPUT
# ============================================================================
city = st.text_input("Enter a city name:", "Atlanta")
days_ahead = st.slider("Days in the future:", 1, 365, 30)
units = st.radio("Temperature units:", ["Fahrenheit", "Celsius"])

# ============================================================================
# MAIN LOGIC
# ============================================================================
if st.button("🔮 Predict Weather"):
    
    # Step 1: Find the city location
    st.info(f"Looking up {city}...")
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_response = requests.get(geo_url)
    geo_data = geo_response.json()
    
    if "results" not in geo_data:
        st.error("City not found! Try a different name.")
        st.stop()
    
    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]
    city_name = geo_data["results"][0]["name"]
    
    # Step 2: Get historical weather data (last 20 years)
    st.info(f"Getting historical data for {city_name}...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=20*365)).strftime("%Y-%m-%d")
    
    unit_param = "fahrenheit" if units == "Fahrenheit" else "celsius"
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean&temperature_unit={unit_param}"
    
    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()
    
    if "daily" not in weather_data:
        st.error("Couldn't get weather data. Try again!")
        st.stop()
    
    temps = weather_data["daily"]["temperature_2m_mean"]
    
    # Calculate statistics
    avg_temp = sum(temps) / len(temps)
    min_temp = min(temps)
    max_temp = max(temps)
    
    unit_symbol = "°F" if units == "Fahrenheit" else "°C"
    
    # Show the data we found
    st.success("Historical data collected!")
    st.write(f"**Average temperature:** {avg_temp:.1f}{unit_symbol}")
    st.write(f"**Lowest recorded:** {min_temp:.1f}{unit_symbol}")
    st.write(f"**Highest recorded:** {max_temp:.1f}{unit_symbol}")
    
    # Step 3: Ask AI to predict the weather
    st.info("Asking AI to predict the weather...")
    
    target_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    prompt = f"""Based on this historical weather data for {city_name}, predict the weather for {target_date}:

Historical Data (20 years):
- Average temperature: {avg_temp:.1f}{unit_symbol}
- Coldest: {min_temp:.1f}{unit_symbol}
- Hottest: {max_temp:.1f}{unit_symbol}

Give me a prediction for {target_date} that includes:
1. Expected temperature range in {unit_symbol}
2. Weather conditions (sunny, cloudy, rainy, etc.)
3. Any other relevant details

Write it as a short weather forecast."""

    # Call Gemini API
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}]
    }
    
    ai_response = requests.post(api_url, json=payload)
    ai_data = ai_response.json()
    
    # Get the prediction text
    try:
        prediction = ai_data["candidates"][0]["content"]["parts"][0]["text"]
        
        st.subheader(f"📅 Weather Prediction for {target_date}")
        st.write(prediction)
        
        # Show sources if available
        if "groundingMetadata" in ai_data["candidates"][0]:
            sources = ai_data["candidates"][0]["groundingMetadata"].get("groundingAttributions", [])
            if sources:
                st.subheader("📚 Sources Used")
                for source in sources:
                    if "web" in source:
                        st.write(f"- [{source['web']['title']}]({source['web']['uri']})")
    
    except:
        st.error("AI couldn't generate a prediction. Try again!")
        st.json(ai_data)  # Show error details
