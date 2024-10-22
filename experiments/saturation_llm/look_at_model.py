import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast

# Configuration parameters
model_name = "EleutherAI/pythia-1b"
tokenizer_path = "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json"

# Load the model and tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
model = GPTNeoXForCausalLM.from_pretrained(model_name)
print(model)
# Set model to evaluation mode
model.eval()

# Dictionary to store the outputs of the internal layers
hook_outputs = {}

# Hook function to capture the output of the internal layers
def hook_fn(module, input, output):
    layer_name = module.__class__.__name__
    
    if isinstance(output, tuple):
        # If the output is a tuple, print the shapes of the elements
        print(f"Captured output from {layer_name}: tuple with {len(output)} elements")
        for i, element in enumerate(output):
            if hasattr(element, 'shape'):
                print(f" - Element {i} shape: {element.shape}")
    else:
        # If the output is a tensor, print its shape
        print(f"Captured output from {layer_name}: shape = {output.shape}")
    
    hook_outputs[layer_name] = output

# Attach hooks to the attention and feed-forward (MLP) layers in each transformer block
for idx, block in enumerate(model.gpt_neox.layers):
    # Hook into the GPTNeoXAttention layer directly
    block.attention.register_forward_hook(hook_fn)
    
    # Hook into the feed-forward (MLP) layer
    block.mlp.register_forward_hook(hook_fn)

# Sample input text
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors='pt')

# Remove token_type_ids from the inputs since the model does not accept it
inputs.pop("token_type_ids", None)

# Perform a forward pass through the model
with torch.no_grad():
    outputs = model(**inputs)

# Print the captured shapes from the hook outputs
print("\n=== Hook Output Summary ===")
for layer_name, output in hook_outputs.items():
    if isinstance(output, tuple):
        print(f"Layer: {layer_name} - Tuple with {len(output)} elements")
    else:
        print(f"Layer: {layer_name}, Output Shape: {output.shape}")
