import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Define the alpaca prompt template
alpaca_prompt = "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"

# Function to load the model and tokenizer with 4-bit quantization
@st.cache_resource
def load_model():
    """Load the base model with 4-bit quantization and the fine-tuned model."""
    st.info("Loading model with 4-bit quantization...")
    
    # Load the base model using 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/meta-llama-3.1-8b-bnb-4bit",  
        load_in_4bit=True,                     
        device_map="auto"  # Automatically maps to GPU if available
    )
    
    # Load the fine-tuned PEFT model
    model = PeftModel.from_pretrained(base_model, "tarek009/my_little_broker")
    
    # Convert to half precision (optional for performance)
    model.half()

    # Optional: Apply PyTorch dynamic optimization
    model = torch.compile(model)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit")
    
    return model, tokenizer

# Function to generate response from the model using the Alpaca prompt format
def generate_response(instruction, input_text, model, tokenizer, max_new_tokens=64):
    """Generate a response from the model using the Alpaca prompt format."""
    # Create the prompt using Alpaca template
    prompt = alpaca_prompt.format(instruction, input_text, "")
    
    # Tokenize the input prompt and move it to GPU if available
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate the output from the model
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    
    # Decode the generated tokens to text
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return response

# Initialize Streamlit app
st.set_page_config(page_title="Little Broker Model Deployment", layout="wide")

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

st.title("🌟 Little Broker Model Deployment with Alpaca Prompt")
st.write("## Generate Responses with Hugging Face Model and 4-bit Quantization")

# Load the model
model, tokenizer = load_model()

# Sidebar for user input
st.sidebar.header("User Input")
instruction = st.sidebar.text_area("Enter your instruction:", height=100)
input_text = st.sidebar.text_area("Enter additional input (optional):", height=100)

# Control for maximum response length
max_length = st.sidebar.slider("Maximum Length of Response", 10, 100, 64)

# Button for generating response
if st.sidebar.button("Generate"):
    if instruction:
        # Generate response
        response = generate_response(instruction, input_text, model, tokenizer, max_new_tokens=max_length)
        st.success("### Generated Response:")
        st.write(response)
    else:
        st.warning("Please enter an instruction!")

# Footer section
st.markdown("---")
st.markdown("### About this App")
st.write(
    "This application uses a Hugging Face model with 4-bit quantization to generate text responses based on user prompts. "
    "It is powered by Streamlit and deployed on Azure."
)
st.write(" 🌐 [GitHub](https://github.com/NaMN21/Little-broker)")
