import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# Define the alpaca prompt template
alpaca_prompt = "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"

# Load the base model with 4-bit quantization using bitsandbytes
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/meta-llama-3.1-8b-bnb-4bit",  # Base model
        load_in_4bit=True,                     # Enable 4-bit quantization
        device_map="auto"                      # Automatically maps the model to the appropriate device (GPU if available)
    )

    # Load the PEFT fine-tuned model
    model = PeftModel.from_pretrained(base_model, "tarek009/my_little_broker")  # Your fine-tuned PEFT model

    # Optional: Convert the model to half-precision for faster inference (fp16)
    model.half()

    # Optional: Use PyTorch 2.0+ dynamic optimization
    model = torch.compile(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit")

    return model, tokenizer

# Streamlit App Interface
st.title("🌟 My Little Broker LLM 🌟")

# Introduce spacing and organize layout with columns
st.write("___")
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Unlock the Power of AI-Powered Real Estate Assistance!</h3>", unsafe_allow_html=True)

# Create a layout for input and button
col1, col2 = st.columns([3, 1])  # Adjust column ratios for layout

# User input for the prompt in the first column
with col1:
    user_prompt = st.text_area("Enter your prompt:", 
                               value="Where can I find the online listing for the building in Greater Cairo / Bait El Watan El Asasy?", 
                               height=150)

# Display button in the second column
with col2:
    st.write("")  # Empty space for alignment
    st.write("")  # More space
    generate_btn = st.button("🚀 Generate Response", key="generate")

# Load the model and tokenizer
model, tokenizer = load_model()

if generate_btn:
    # Define your prompt using the alpaca format
    prompt = alpaca_prompt.format(user_prompt, "", "")

    # Tokenize the input prompt and move it to GPU if available
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate the output (text generation)
    with st.spinner("🤖 Generating response..."):
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode and clean the generated text
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    generated_text = generated_text.split("### Response:")[1].strip()

    # Display the result in a centered section
    st.write("___")
    st.markdown("<h4 style='text-align: center;'>Generated Response:</h4>", unsafe_allow_html=True)
    st.success(generated_text)

# Footer to make the UI look polished
st.write("___")
st.markdown("<h6 style='text-align: center;'>© 2024 My Little Broker LLM. Powered by Streamlit and Hugging Face.</h6>", unsafe_allow_html=True)
