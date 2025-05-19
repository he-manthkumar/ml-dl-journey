import streamlit as st
from rag_pipeline import process_query

# Set page config
st.set_page_config(
    page_title="Indian Constitution RAG Bot",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light theme
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }
    
    /* Title styling */
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        font-size: 1.2rem;
        color: #34495e;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #2ecc71;
    }
    
    /* Input area styling */
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
        background-color: #3498db;
        color: white;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
        height: 35px;
        margin-top: 25px;
        line-height: 1;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    /* Loading spinner styling */
    .stSpinner>div {
        border-color: #3498db;
    }
    
    /* Strong text in messages */
    .chat-message strong {
        color: #2c3e50;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h2 style='color: #2c3e50;'>📜 About</h2>
            <p style='color: #34495e;'>This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about the Indian Constitution.</p>
            <hr style='border-color: #e0e0e0;'>
            <h3 style='color: #2c3e50;'>💡 Sample Questions</h3>
            <ul style='text-align: left; color: #34495e;'>
                <li>Who was the Chairman of the Drafting Committee?</li>
                <li>What are Fundamental Duties?</li>
                <li>How many languages are officially recognized?</li>
                <li>What is the Panchayati Raj system?</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("""
    <div class='title-text'>📜 Indian Constitution RAG Bot</div>
    <div class='subtitle-text'>Ask any question about the Indian Constitution and I'll try to help!</div>
""", unsafe_allow_html=True)

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class='chat-message user-message'>
                    <div style='flex-grow: 1;'>
                        <strong>You:</strong><br>
                        {message["content"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <div style='flex-grow: 1;'>
                        <strong>Assistant:</strong><br>
                        {message["content"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Input area
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input("Ask a question about the Indian Constitution...", key="user_input")
with col2:
    send_button = st.button("Send", use_container_width=True)

# Process input
if (send_button and user_input) or (user_input and st.session_state.get("last_input") != user_input):
    # Store the current input
    st.session_state.last_input = user_input
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with chat_container:
        st.markdown(f"""
            <div class='chat-message user-message'>
                <div style='flex-grow: 1;'>
                    <strong>You:</strong><br>
                    {user_input}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Get bot response with loading animation
    with st.spinner("Thinking..."):
        response = process_query(user_input)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response
    with chat_container:
        st.markdown(f"""
            <div class='chat-message assistant-message'>
                <div style='flex-grow: 1;'>
                    <strong>Assistant:</strong><br>
                    {response}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Force a rerun to clear the input
    st.rerun()

# Add footer
st.markdown("""
    <div class='footer'>
        <p>Built with ❤️ using Streamlit and RAG</p>
    </div>
""", unsafe_allow_html=True)
