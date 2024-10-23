import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def load_model():
    config = PeftConfig.from_pretrained("tarek009/my_little_broker")
    base_model = AutoModelForCausalLM.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit")
    model = PeftModel.from_pretrained(base_model, "tarek009/my_little_broker")
    return model

def generate_response(prompt, model, max_length=50):
    inputs = model.tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main(prompt, max_length):
    model = load_model()
    response = generate_response(prompt, model, max_length)
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a response from the model.')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to generate a response for.')
    parser.add_argument('--max_length', type=int, default=50, help='The maximum length of the generated response.')
    args = parser.parse_args()

    main(args.prompt, args.max_length)
