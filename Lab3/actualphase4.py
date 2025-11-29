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
    st.write("Ask me about weather in any city!")
    st.write("Example: *What's the weather in Paris for the last week?*")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about weather..."):
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
                # Create Gemini model
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                
                # Build conversation history for context
                conversation = []
                for msg in st.session_state.messages:
                    conversation.append(f"{msg['role']}: {msg['content']}")
                
                # System prompt
                system_context = """You are a helpful weather assistant. When users ask about weather:
1. Extract the city name and date range from their question
2. Use this information to fetch weather data from the Open-Meteo API
3. Provide helpful answers based on the data

For weather data fetching:
- Use Open-Meteo Geocoding API: https://geocoding-api.open-meteo.com/v1/search?name={city}
- Use Open-Meteo Archive API: https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean

If the user asks about weather, explain what you found and provide insights.
If you need to fetch data, describe what data you would fetch and provide a helpful response.

Current date is """ + str(datetime.date.today())
                
                full_prompt = system_context + "\n\n" + "\n".join(conversation[-5:])
                
                # Check if user is asking about weather
                if any(word in prompt.lower() for word in ['weather', 'temperature', 'temp', 'hot', 'cold', 'warm']):
                    # Try to extract city and dates
                    extract_prompt = f"""Extract the city name and date range from this question: "{prompt}"
                    
                    Return in this format:
                    City: [city name]
                    Start Date: [YYYY-MM-DD or 'recent' or 'today']
                    End Date: [YYYY-MM-DD or 'recent' or 'today']
                    
                    If no specific dates mentioned, use the last 7 days."""
                    
                    extraction = model.generate_content(extract_prompt)
                    extraction_text = extraction.text
                    
                    # Simple parsing to get city
                    city = None
                    for line in extraction_text.split('\n'):
                        if 'City:' in line or 'city:' in line:
                            city = line.split(':')[1].strip()
                            break
                    
                    if city and city.lower() not in ['none', 'not specified', 'n/a']:
                        try:
                            # Fetch weather data
                            # Geocoding
                            url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
                            response = requests.get(url, timeout=10)
                            data = response.json()
                            
                            if "results" in data and len(data["results"]) > 0:
                                # Get first result
                                result = data["results"][0]
                                lat = result["latitude"]
                                lon = result["longitude"]
                                city_name = result["name"]
                                
                                # Default to last 7 days
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
                                    
                                    weather_context = f"""
Weather data for {city_name} from {start_date} to {end_date}:
- Average temperature: {avg_temp}°F
- Highest temperature: {max_temp}°F
- Lowest temperature: {min_temp}°F
- Daily temperatures: {list(zip(dates, temps))}

User question: {prompt}

Provide a helpful, conversational answer about this weather data."""
                                    
                                    response = model.generate_content(weather_context)
                                    assistant_response = response.text
                                else:
                                    assistant_response = f"I found {city_name} but couldn't get weather data for that time period. Try a different date range?"
                            else:
                                assistant_response = f"I couldn't find weather data for '{city}'. Could you check the city name?"
                        
                        except Exception as e:
                            assistant_response = f"I had trouble fetching weather data: {str(e)}. Could you try rephrasing your question?"
                    else:
                        # No city found, general response
                        response = model.generate_content(full_prompt)
                        assistant_response = response.text
                else:
                    # General conversation
                    response = model.generate_content(full_prompt)
                    assistant_response = response.text
                
                st.write(assistant_response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
