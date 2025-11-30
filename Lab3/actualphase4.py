import streamlit as st
import datetime
import requests
import google.generativeai as genai

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Weather Chat Assistant", page_icon="🌤️")
st.title("🌤️ Weather Chat Assistant")

# ============================================================================
# CONSTANTS
# ============================================================================
MODEL_NAME = "gemini-2.0-flash-exp"
WEATHER_KEYWORDS = ['weather', 'temperature', 'temp', 'hot', 'cold', 'warm', 'climate']
GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"

# ============================================================================
# API KEY CONFIGURATION
# ============================================================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except (KeyError, FileNotFoundError):
    st.error("⚠️ API key not found. Please configure GEMINI_API_KEY in your secrets.")
    st.info("For local development, create `.streamlit/secrets.toml` with your API key.")
    st.stop()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_response_text(response):
    """
    Safely extract text from Gemini response with error handling.
    
    Args:
        response: Gemini API response object
        
    Returns:
        str: Extracted text or error message
    """
    try:
        # Check if response has valid candidates
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            # Check finish reason (1 = STOP means successful completion)
            if candidate.finish_reason != 1:
                finish_reason_map = {
                    2: "MAX_TOKENS",
                    3: "SAFETY",
                    4: "RECITATION",
                    5: "OTHER"
                }
                reason = finish_reason_map.get(candidate.finish_reason, "UNKNOWN")
                return f"⚠️ Response was blocked or incomplete. Reason: {reason}. Please try rephrasing your question."
            
            # Extract text from parts
            if candidate.content and candidate.content.parts:
                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                if text_parts:
                    return ''.join(text_parts)
        
        return "⚠️ No response generated. Please try rephrasing your question."
    
    except Exception as e:
        return f"⚠️ Error extracting response: {str(e)}"


def get_safety_settings():
    """
    Configure safety settings for Gemini API.
    
    Returns:
        list: Safety settings configuration
    """
    return [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]


def extract_city_name(model, prompt, safety_settings):
    """
    Extract city name from user prompt using Gemini.
    
    Args:
        model: Gemini model instance
        prompt: User's original prompt
        safety_settings: Safety configuration
        
    Returns:
        str: Extracted and cleaned city name
    """
    extract_prompt = f'From this question, extract ONLY the city name. Return just the city name, nothing else: "{prompt}"'
    extraction = model.generate_content(extract_prompt, safety_settings=safety_settings)
    city = get_response_text(extraction).strip()
    
    # Clean up city name
    if not city.startswith("⚠️"):
        city = city.replace('"', '').replace("'", "").split(',')[0].split('.')[0].strip()
    
    return city


def fetch_location_data(city):
    """
    Fetch geocoding data for a city.
    
    Args:
        city: City name to search for
        
    Returns:
        dict: Location data or None if not found
    """
    try:
        url = f"{GEOCODING_API}?name={city}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]
        return None
    except Exception as e:
        raise Exception(f"Geocoding error: {str(e)}")


def fetch_weather_data(lat, lon, start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date for weather data
        end_date: End date for weather data
        
    Returns:
        dict: Weather data or None if not found
    """
    try:
        url = f"{WEATHER_API}?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&temperature_unit=fahrenheit"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "daily" in data:
            return data["daily"]
        return None
    except Exception as e:
        raise Exception(f"Weather API error: {str(e)}")


def analyze_weather_trend(temps):
    """
    Analyze temperature trend from recent data.
    
    Args:
        temps: List of temperatures
        
    Returns:
        str: Trend description (warming, cooling, or stable)
    """
    if len(temps) < 3:
        return "stable"
    
    recent_temps = temps[-3:]
    temp_change = recent_temps[-1] - recent_temps[0]
    
    if temp_change > 3:
        return "warming"
    elif temp_change < -3:
        return "cooling"
    return "stable"


def create_weather_context(city_name, start_date, end_date, weather_data, prompt):
    """
    Create a comprehensive weather context for the AI model.
    
    Args:
        city_name: Name of the city
        start_date: Start date of weather data
        end_date: End date of weather data
        weather_data: Daily weather data
        prompt: User's original question
        
    Returns:
        str: Formatted weather context for AI
    """
    temps = weather_data["temperature_2m_mean"]
    temp_max = weather_data["temperature_2m_max"]
    temp_min = weather_data["temperature_2m_min"]
    dates = weather_data["time"]
    
    # Calculate statistics
    avg_temp = round(sum(temps) / len(temps), 1)
    max_temp = max(temp_max)
    min_temp = min(temp_min)
    
    # Analyze trend and predict tomorrow
    recent_trend = analyze_weather_trend(temps)
    recent_temps = temps[-3:]
    predicted_temp = round(sum(recent_temps) / len(recent_temps), 1)
    tomorrow = end_date + datetime.timedelta(days=1)
    
    # Create daily summary
    daily_summary = "\n".join([f"{dates[i]}: {temps[i]}°F" for i in range(len(dates))])
    
    # Build context string
    context = f"""Weather data for {city_name} from {start_date} to {end_date}:
- Average temperature over past week: {avg_temp}°F
- Highest temperature: {max_temp}°F  
- Lowest temperature: {min_temp}°F
- Recent trend: {recent_trend}
- Last 3 days: {recent_temps[-3]}°F, {recent_temps[-2]}°F, {recent_temps[-1]}°F

Daily temperatures:
{daily_summary}

PREDICTION FOR TOMORROW ({tomorrow}):
Based on the recent trend, tomorrow's temperature is predicted to be around {predicted_temp}°F (trend: {recent_trend}).

User question: {prompt}

IMPORTANT: Use the predicted temperature for tomorrow to answer the user's question about future activities. Be confident in your prediction based on the data trend. Suggest specific outdoor activities appropriate for the predicted temperature range. Don't say you don't have tomorrow's data - you have a prediction based on the trend."""
    
    return context


def handle_weather_query(model, prompt, safety_settings):
    """
    Handle weather-related queries by fetching data and generating response.
    
    Args:
        model: Gemini model instance
        prompt: User's question
        safety_settings: Safety configuration
        
    Returns:
        str: AI-generated response
    """
    # Extract city name
    city = extract_city_name(model, prompt, safety_settings)
    
    # Check if extraction was successful
    if city.startswith("⚠️") or not city or len(city) >= 50:
        # Extraction failed, try general response
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return get_response_text(response)
    
    try:
        # Fetch location data
        location = fetch_location_data(city)
        if not location:
            return f"I couldn't find a city called '{city}'. Could you try a different city name?"
        
        city_name = location["name"]
        lat = location["latitude"]
        lon = location["longitude"]
        
        # Calculate date range (last 7 days)
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=7)
        
        # Fetch weather data
        weather_data = fetch_weather_data(lat, lon, start_date, end_date)
        if not weather_data:
            return f"I found {city_name} but couldn't get weather data. Try asking about a different time period!"
        
        # Create context and generate response
        weather_context = create_weather_context(city_name, start_date, end_date, weather_data, prompt)
        response = model.generate_content(weather_context, safety_settings=safety_settings)
        return get_response_text(response)
        
    except Exception as e:
        return f"I had trouble getting weather data: {str(e)}"


def is_weather_query(prompt):
    """
    Check if the prompt is asking about weather.
    
    Args:
        prompt: User's question
        
    Returns:
        bool: True if weather-related query
    """
    return any(word in prompt.lower() for word in WEATHER_KEYWORDS)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# DISPLAY CHAT HISTORY
# ============================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
if prompt := st.chat_input("Ask me about weather in any city..."):
    # Check API key
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Initialize model and settings
                model = genai.GenerativeModel(MODEL_NAME)
                safety_settings = get_safety_settings()
                
                # Route to appropriate handler
                if is_weather_query(prompt):
                    assistant_response = handle_weather_query(model, prompt, safety_settings)
                else:
                    # General conversation
                    response = model.generate_content(prompt, safety_settings=safety_settings)
                    assistant_response = get_response_text(response)
                
                # Display and save response
                st.write(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
