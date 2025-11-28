import streamlit as st
import requests
import json
import time
import re # For simple keyword checking

# --- Configuration & Gemini API Setup ---
# NOTE: Replace with your actual Gemini API key
API_KEY = "" 
MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash-preview-09-2025"

st.title("☀️ Gemini Weather Synthesizer")
st.caption("Built using the simple chat and streaming techniques from the Streamlit tutorial.")

# Check for API Key
if not API_KEY:
    st.error("⚠️ Gemini API_KEY is missing. Please update the API_KEY variable.")
    st.stop()

# --- Weather Tool Definitions (Copied from previous file) ---

def get_population(city_data):
    """Helper function to safely retrieve the population for city comparison."""
    return city_data.get("population", 0)

def geocode_city(city):
    """Fetches latitude and longitude for a city."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    
    # Simplified logic without try/except blocks, as requested.
    # This will raise a requests.exceptions.RequestException or json.JSONDecodeError 
    # if the network fails or the response is invalid.
    response = requests.get(url, timeout=10)
    response.raise_for_status() 
    data = response.json()
    
    if "results" not in data or len(data["results"]) == 0:
        return None, None
    
    best_match = None
    max_population = -1 

    for city_data in data["results"]:
        current_population = get_population(city_data)
        if current_population > max_population:
            max_population = current_population
            best_match = city_data
            
    if best_match is None:
        return None, None
        
    return best_match.get("latitude"), best_match.get("longitude")

def get_current_and_forecast_weather(city: str, units: str = 'celsius') -> str:
    """Provides the current temperature and a summary of the 7-day weather forecast."""
    if units not in ['celsius', 'fahrenheit']:
        return json.dumps({"error": "Invalid units. Must be 'celsius' or 'fahrenheit'."})

    lat, lon = geocode_city(city)
    if lat is None:
        return json.dumps({"error": f"Could not find coordinates for city: {city}"})

    temp_unit = "temperature_unit=fahrenheit" if units == "fahrenheit" else ""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min&current=temperature_2m,wind_speed_10m&timezone=auto&{temp_unit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        daily = data.get('daily', {})
        
        unit_symbol = "°F" if units == "fahrenheit" else "°C"

        forecasts = []
        for i in range(min(7, len(daily.get('time', [])))):
            forecasts.append({
                "date": daily['time'][i],
                "max_temp": daily['temperature_2m_max'][i],
                "min_temp": daily['temperature_2m_min'][i],
            })

        # Return structured data for the LLM to synthesize
        return json.dumps({
            "city": city,
            "units": unit_symbol,
            "current_conditions": {
                "temperature": current.get('temperature_2m'),
                "wind_speed": current.get('wind_speed_10m'),
            },
            "seven_day_forecast": forecasts
        })

    except Exception as e:
        return json.dumps({"error": f"Error fetching weather data: {e}"})

# --- LLM Integration Logic ---

# Function to simulate streaming a non-streamed response (mimics tutorial page 9)
def stream_response_text(full_response_text):
    message_placeholder = st.empty()
    full_text = ""
    # Use the same sleep timing as the tutorial (0.05s)
    for chunk in full_response_text.split():
        full_text += chunk + " "
        time.sleep(0.05)
        # Display the text with a cursor at the end
        message_placeholder.markdown(full_text + "▌") 
    message_placeholder.markdown(full_text)
    return full_text

def call_gemini_api(messages):
    """Makes a direct, non-streaming call to the Gemini API for text generation."""
    full_url = f"{MODEL_URL}?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    # Convert Streamlit messages history to the Gemini API format
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} 
                for m in messages]

    payload = {
        "contents": contents,
        # Using Google Search grounding tool to provide up-to-date general info
        "config": {"tools": [{"google_search": {}}]}
    }

    try:
        with st.spinner("Thinking..."):
            response = requests.post(full_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status() 
            result = response.json()
        
        candidate = result.get('candidates', [{}])[0]
        text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No response generated.')
        
        # NOTE: Sources are not extracted here to keep complexity low, matching the tutorial's focus.
        return text

    except requests.exceptions.RequestException as e:
        return f"Error during Gemini API call: {e}"
    except json.JSONDecodeError:
        return "Error: Could not decode response from Gemini API."

# --- Streamlit Application Layout (Following Tutorial Structure) ---

# Initialize chat history (as per tutorial)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can answer general questions and provide weather data. Try asking for the weather in Paris!"}
    ]

# Display chat messages from history on app rerun (as per tutorial)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input (as per tutorial)
if prompt := st.chat_input("Ask a question or check the weather..."):
    
    # 1. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Check for Weather Keywords (Hybrid Logic)
    city_match = re.search(r"weather in ([\w\s]+)", prompt, re.IGNORECASE)
    
    if city_match:
        city = city_match.group(1).strip()
        st.info(f"Detected weather query for **{city}**. Calling weather API directly...")
        
        # Execute the weather tool locally
        # Note: This call may now fail the application if a network error occurs.
        try:
            weather_data_json = get_current_and_forecast_weather(city)
        except Exception as e:
            weather_data_json = json.dumps({"error": f"A network or JSON parsing error occurred during geocoding: {e}"})

        # Construct the final prompt to the LLM for synthesis
        synthesis_prompt = f"""
        User Query: "{prompt}"
        
        Weather Data (JSON):
        {weather_data_json}
        
        Synthesize the above weather data into a friendly, natural language response based on the user's query. Do not output the JSON directly. If there was an error fetching data, report the error nicely.
        """
        
        # Use the synthesis prompt for the next LLM call
        llm_input_content = synthesis_prompt
        
        # We need to manually add the weather data to the history for context, 
        # but hide it from the user by not rerunning yet.
        st.session_state.messages.append({"role": "weather_data", "content": f"Weather API returned: {weather_data_json}"})

    else:
        # General chat - use the user's prompt directly
        llm_input_content = prompt

    # 3. Get LLM Response
    
    # Since we modified the input content, we temporarily add it to a list for the API call
    # and then revert to the history for the final save.
    
    # The messages sent to the API are the full history PLUS the temporary synthesis prompt
    messages_for_api = st.session_state.messages + [{"role": "user", "content": llm_input_content}]

    gemini_response_text = call_gemini_api(messages_for_api)

    # 4. Display Assistant Response (with streaming simulation, per tutorial)
    with st.chat_message("assistant"):
        final_response = stream_response_text(gemini_response_text)
    
    # 5. Add Assistant Response to History (as per tutorial)
    # If the weather query was made, we need to ensure only the user/assistant messages are visible.
    if 'weather_data' in st.session_state.messages[-1]['role']:
        st.session_state.messages.pop() # Remove the hidden weather_data entry
    
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    
    # Rerun to clear the temporary st.info and ensure final state
    st.rerun()
