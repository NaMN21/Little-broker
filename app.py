import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Function to load the model and tokenizer
@st.cache_resource
def load_model():
    """Load the Hugging Face model."""
    st.info("Loading model...")
    config = PeftConfig.from_pretrained("tarek009/my_little_broker")
    base_model =AutoModelForCausalLM.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit", device_map="cpu", load_in_8bit=False)
    model = PeftModel.from_pretrained(base_model, "tarek009/my_little_broker")
    tokenizer = AutoTokenizer.from_pretrained("tarek009/my_little_broker")
    return model, tokenizer

# Function to generate response from the model
def generate_response(prompt, model, tokenizer, max_length=50):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="Little Broker Model", layout="wide")

# Custom CSS for background and button style
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8; /* Light background color */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        border: none; /* Remove borders */
        padding: 10px 20px; /* Add some padding */
        text-align: center; /* Center the text */
        text-decoration: none; /* No underline */
        display: inline-block; /* Make it inline-block */
        font-size: 16px; /* Increase font size */
        margin: 4px 2px; /* Add some margin */
        cursor: pointer; /* Pointer cursor on hover */
        border-radius: 5px; /* Rounded corners */
        transition: background-color 0.3s; /* Smooth transition */
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌟 Little Broker Model Deployment")
st.write("## Generate Responses with Hugging Face Model")

# Load the model
model, tokenizer = load_model()

# Sidebar for user input
st.sidebar.header("User Input")
user_input = st.sidebar.text_area("Enter your prompt here:", height=200)

# Control for maximum response length
max_length = st.sidebar.slider("Maximum Length of Response", 10, 100, 50)

# Button for generating response
if st.sidebar.button("Generate"):
    if user_input:
        # Generate response
        response = generate_response(user_input, model, tokenizer, max_length)
        st.success("### Generated Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt!")

# Footer section
st.markdown("---")
st.markdown("### About this App")
st.write(
    "This application uses a Hugging Face model to generate text responses based on user prompts. "
    "It is powered by Streamlit and deployed on Azure."
)
st.write(" 🌐 [GitHub](https://github.com/NaMN21/Little-broker)")
