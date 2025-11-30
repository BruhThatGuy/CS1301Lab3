import streamlit as st
import datetime
import requests
import google.generativeai as genai

st.set_page_config(page_title="Weather Chat Assistant", page_icon="🌤️")

st.title("🌤️ Weather Chat Assistant")

# Sidebar for API key and settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key:", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Key configured!")
    else:
        st.info("Get your API key from https://makersuite.google.com/app/apikey")
    
    st.write("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about weather in any city..."):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Create Gemini model with user-specified name
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                # Check if user is asking about weather
                if any(word in prompt.lower() for word in ['weather', 'temperature', 'temp', 'hot', 'cold', 'warm', 'climate']):
                    # Try to extract city
                    extract_prompt = f"""From this question, extract ONLY the city name. Return just the city name, nothing else: "{prompt}" """
                    
                    extraction = model.generate_content(extract_prompt)
                    city = extraction.text.strip()
                    
                    # Clean up the city name
                    city = city.replace('"', '').replace("'", "").split(',')[0].split('.')[0].strip()
                    
                    if city and len(city) > 0 and len(city) < 50:
                        try:
                            # Fetch weather data
                            url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
                            response = requests.get(url, timeout=10)
                            data = response.json()
                            
                            if "results" in data and len(data["results"]) > 0:
                                result = data["results"][0]
                                lat = result["latitude"]
                                lon = result["longitude"]
                                city_name = result["name"]
                                
                                # Last 7 days
                                end_date = datetime.date.today()
                                start_date = end_date - datetime.timedelta(days=7)
                                
                                # Fetch weather
                                weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&temperature_unit=fahrenheit"
                                weather_response = requests.get(weather_url, timeout=10)
                                weather_data = weather_response.json()
                                
                                if "daily" in weather_data:
                                    temps = weather_data["daily"]["temperature_2m_mean"]
                                    temp_max = weather_data["daily"]["temperature_2m_max"]
                                    temp_min = weather_data["daily"]["temperature_2m_min"]
                                    dates = weather_data["daily"]["time"]
                                    
                                    avg_temp = round(sum(temps) / len(temps), 1)
                                    max_temp = max(temp_max)
                                    min_temp = min(temp_min)
                                    
                                    # Create a summary
                                    daily_summary = "\n".join([f"{dates[i]}: {temps[i]}F" for i in range(len(dates))])
                                    
                                    weather_context = f"""Weather data for {city_name} from {start_date} to {end_date}:
- Average temperature: {avg_temp} degrees Fahrenheit
- Highest temperature: {max_temp} degrees Fahrenheit  
- Lowest temperature: {min_temp} degrees Fahrenheit

Daily temperatures:
{daily_summary}

User question: {prompt}

Provide a helpful, friendly, conversational answer about this weather data."""
                                    
                                    response = model.generate_content(weather_context)
                                    assistant_response = response.text
                                else:
                                    assistant_response = f"I found {city_name} but couldn't get weather data. Try asking about a different time period!"
                            else:
                                assistant_response = f"I couldn't find a city called '{city}'. Could you try a different city name?"
                        
                        except Exception as e:
                            assistant_response = f"I had trouble getting weather data: {str(e)}"
                    else:
                        # General weather question without specific city
                        response = model.generate_content(prompt)
                        assistant_response = response.text
                else:
                    # General conversation
                    response = model.generate_content(prompt)
                    assistant_response = response.text
                
                st.write(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
