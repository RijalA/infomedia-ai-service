import requests
import streamlit as st
import time
import json

url = "http://127.0.0.1:8000"

vendor = "posindo"

st.title(f"Demo for {vendor.upper()}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = requests.post(
        url=f"{url}/agents/ask",
        json={"vendor": vendor, "query": prompt}
        ).json()
    response = response["answer"]
    response_msg = json.loads(response)["response_chat"]
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response_msg.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        # st.json(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_msg})