import streamlit as st
import requests
from Helpers.configs import get_settings
import time
import mimetypes
app_setting = get_settings()



def get_message(user_input):
    response = requests.post(app_setting.FASTAPI_URL+'/chat',  params={"query": user_input})
    bot_message = response.json().get("query")
    return bot_message

def response_generator(bot_message):
    for word in bot_message.split("\n"):
        yield word + " \n"
        time.sleep(0.080)

def upload_file(file):
    mime_type, _ = mimetypes.guess_type(file.name)
    files = {
        "file": (file.name, file.getvalue(), mime_type)
    }
    response = requests.post(app_setting.FASTAPI_URL+'/upload',  files=files)
    output = response.json().get("query")
    return output , response.status_code

st.title("Chat With Your Documents")

response = requests.get(app_setting.FASTAPI_URL)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": response.json().get('query')}
    ]


    # Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input := st.chat_input("Type your message"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("The assistant is thinking..."):
        bot_message = get_message(user_input)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        output = st.write_stream(response_generator(bot_message))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output})

