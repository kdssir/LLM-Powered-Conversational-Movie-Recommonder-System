import streamlit as st
import requests

st.set_page_config(page_title="Conversational Recommender", page_icon="🎬")

st.title("🎬 Conversational Movie Recommender")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input field
user_input = st.chat_input("Tell me what kind of movie you're in the mood for...")

if user_input:
    # Add user message to chat history 
    print(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Send to FastAPI backend
    try:
        response = requests.post(
            "http://localhost:8000/recommend", 
            json={"message": user_input}
        )
        data = response.json()
        
        if "result" in data:
            recs = data["recommended_movies"]
            rec_text = ""
            rec_text += "\n".join([f"🎥 **Movie**: *{rec['title']}* || **Genre**: {rec['genres']}"+"\n" for rec in recs])
        else:
            rec_text = data.get("message", "Sorry, could not find any recommendations.")
    except Exception as e:
        rec_text = f"🚨 Error: {str(e)}"

    # Add bot message
    st.session_state.messages.append({"role": "assistant", "content": rec_text})

    # Display bot message
    with st.chat_message("assistant"):
        st.markdown(rec_text)
