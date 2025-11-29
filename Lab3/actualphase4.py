import streamlit as st
import datetime
import requests
import google.generativeai as genai

st.set_page_config(page_title="Weather Chat Assistant", page_icon="🌤️")

st.header("🌤️ Weather Data Chat Assistant")
st.write("Fetch weather data and ask questions about it!")

# Gemini API key input
if "api_key" not in st.session_state:
    st.session_state.api_key = "AIzaSyBg7BL-ACkEFFkSHjTxXk_trTOJu1vON5I"

api_key = st.text_input("Enter your Gemini API Key:", type="password", value=st.session_state.api_key)

if api_key:
    st.session_state.api_key = api_key
    genai.configure(api_key=api_key)
else:
    st.info("Please enter your Gemini API key to continue. Get one at https://makersuite.google.com/app/apikey")
    st.stop()

st.write("---")

# Input fields
col1, col2 = st.columns(2)
with col1:
    city = st.text_input("City:", "Atlanta")
    start = st.date_input("Start Date:", value=datetime.date(2024, 1, 1), 
                          min_value=datetime.date(1995, 1, 1), 
                          max_value=datetime.date(2024, 10, 31))

with col2:
    units = st.selectbox("Temperature Units:", ["Fahrenheit", "Celsius"])
    end = st.date_input("End Date:", value=datetime.date(2024, 10, 31),
                        min_value=datetime.date(1995, 1, 1), 
                        max_value=datetime.date(2025, 10, 31))

# Initialize session state
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "city_info" not in st.session_state:
    st.session_state.city_info = {}

# Fetch weather data
if st.button("Fetch Weather Data"):
    if end < start:
        st.error("End date must be after start date.")
        st.stop()
    
    try:
        with st.spinner("Fetching weather data..."):
            # Geocoding
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
            response = requests.get(url)
            data = response.json()
            
            if "results" not in data or len(data["results"]) == 0:
                st.error("City not found. Check spelling or try a different city.")
                st.stop()
            
            # Get coordinates of largest city
            pophigh = 0
            lat, long = 0, 0
            city_name = city
            for c in data["results"]:
                if "population" in c and c["population"] > pophigh:
                    pophigh = c["population"]
                    lat = c["latitude"]
                    long = c["longitude"]
                    city_name = c["name"]
            
            # Fetch weather data
            unit_param = units.lower()
            if unit_param == "fahrenheit":
                url2 = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&hourly=temperature_2m&temperature_unit={unit_param}"
            else:
                url2 = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&hourly=temperature_2m"
            
            response2 = requests.get(url2)
            weather_data = response2.json()
            
            if "daily" not in weather_data or "temperature_2m_mean" not in weather_data["daily"]:
                st.error("No temperature data available for this time frame.")
                st.stop()
            
            # Store data in session state
            st.session_state.weather_data = weather_data
            st.session_state.city_info = {
                "name": city_name,
                "lat": lat,
                "long": long,
                "start_date": str(start),
                "end_date": str(end),
                "units": units
            }
            st.session_state.messages = []  # Clear chat history on new data fetch
            
            st.success(f"Weather data fetched for {city_name}! You can now ask questions.")
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")

st.write("---")

# Chat interface
if st.session_state.weather_data:
    st.subheader(f"💬 Chat about {st.session_state.city_info['name']} Weather")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the weather data..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Prepare context for Gemini
        weather_data = st.session_state.weather_data
        city_info = st.session_state.city_info
        
        dates = weather_data["daily"]["time"]
        temps_mean = weather_data["daily"]["temperature_2m_mean"]
        temps_max = weather_data["daily"]["temperature_2m_max"]
        temps_min = weather_data["daily"]["temperature_2m_min"]
        
        avg_temp = round(sum(temps_mean) / len(temps_mean), 1)
        max_temp = max(temps_max)
        min_temp = min(temps_min)
        
        context = f"""
You are a helpful weather assistant. Answer questions about the following weather data:

City: {city_info['name']}
Location: Latitude {city_info['lat']}, Longitude {city_info['long']}
Date Range: {city_info['start_date']} to {city_info['end_date']}
Temperature Units: {city_info['units']}

Summary Statistics:
- Average Temperature: {avg_temp}°
- Highest Temperature: {max_temp}°
- Lowest Temperature: {min_temp}°
- Number of days: {len(dates)}

Daily Data (Date, Mean Temp, Max Temp, Min Temp):
"""
        
        # Add sample of daily data (limit to avoid token limits)
        sample_size = min(30, len(dates))
        for i in range(0, len(dates), max(1, len(dates) // sample_size)):
            context += f"\n{dates[i]}: Mean={temps_mean[i]}°, Max={temps_max[i]}°, Min={temps_min[i]}°"
        
        context += f"\n\nUser Question: {prompt}\n\nProvide a helpful, concise answer based on this data."
        
        # Generate response with Gemini
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(context)
                    assistant_response = response.text
                    st.write(assistant_response)
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error generating response: {e}")
else:
    st.info("👆 Fetch weather data first to start chatting!")
