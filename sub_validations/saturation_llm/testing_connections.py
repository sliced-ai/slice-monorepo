import torch
import random
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, PreTrainedTokenizerFast
from typing import Optional, Tuple, Union

# Configuration parameters
model_name = "EleutherAI/pythia-70m-deduped"
tokenizer_path = "/workspace/slice-monorepo/sub_validations/cl_scaling/20B_tokenizer.json"

class CustomGPTNeoXWithJumpConnections(GPTNeoXForCausalLM):
    def __init__(self, config: GPTNeoXConfig):
        super().__init__(config)
        self.jump_connections = []
        self.jump_weight = 0.1  # Weight for jump connections

    def create_jump_connections(self, num_connections=50):
        num_layers = len(self.gpt_neox.layers)
        for _ in range(num_connections):
            source_layer = random.randint(0, num_layers - 2)
            target_layer = random.randint(source_layer + 1, num_layers - 1)
            source_neuron = random.randint(0, self.config.hidden_size - 1)
            target_neuron = random.randint(0, self.config.hidden_size - 1)
            self.jump_connections.append((source_layer, target_layer, source_neuron, target_neuron))
            print(f"Created jump connection: Layer {source_layer} (Neuron {source_neuron}) -> Layer {target_layer} (Neuron {target_neuron})")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        output_hidden_states = True  # Force hidden states output

        # Initial forward pass to get embeddings
        hidden_states = self.gpt_neox.embed_in(input_ids)
        all_hidden_states = [hidden_states]

        # Apply transformer layers with integrated jump connections
        for i, layer in enumerate(self.gpt_neox.layers):
            # Collect jump connection inputs for this layer
            jump_inputs = torch.zeros_like(hidden_states)
            for source_layer, target_layer, source_neuron, target_neuron in self.jump_connections:
                if target_layer == i and source_layer < i:
                    source_activation = all_hidden_states[source_layer + 1][:, :, source_neuron]
                    jump_inputs[:, :, target_neuron] += source_activation

            # Integrate jump inputs into the layer's computation
            hidden_states_before_jump = hidden_states.clone()
            hidden_states = (1 - self.jump_weight) * hidden_states + self.jump_weight * jump_inputs

            # Process the current layer
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)

            # Print detailed information about jump connections for this layer
            for source_layer, target_layer, source_neuron, target_neuron in self.jump_connections:
                if target_layer == i and source_layer < i:
                    source_value = all_hidden_states[source_layer + 1][0, 0, source_neuron].item()
                    target_before = hidden_states_before_jump[0, 0, target_neuron].item()
                    target_after = hidden_states[0, 0, target_neuron].item()
                    print(f"\nJump connection: Layer {source_layer} (Neuron {source_neuron}) -> Layer {target_layer} (Neuron {target_neuron})")
                    print(f"  Source neuron value: {source_value:.6f}")
                    print(f"  Target neuron value before jump: {target_before:.6f}")
                    print(f"  Target neuron value after jump: {target_after:.6f}")
                    print(f"  Change due to jump: {target_after - target_before:.6f}")

        # Final layer norm
        hidden_states = self.gpt_neox.final_layer_norm(hidden_states)

        # Compute logits
        lm_logits = self.embed_out(hidden_states)

        return {
            'logits': lm_logits,
            'hidden_states': tuple(all_hidden_states),
            'last_hidden_state': hidden_states,
        }

# Load the model and tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
config = GPTNeoXConfig.from_pretrained(model_name)
model = CustomGPTNeoXWithJumpConnections.from_pretrained(model_name, config=config)

# Create jump connections
model.create_jump_connections(num_connections=5)  # You can adjust the number of connections

# Set model to evaluation mode
model.eval()

# Sample input text
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors='pt')

# Remove token_type_ids from the inputs if the model does not accept it
inputs.pop("token_type_ids", None)

# Function to run inference and print results
def run_inference_and_print():
    print("\n--- Running inference with integrated jump connections ---")
    
    # Perform inference without jump connections
    with torch.no_grad():
        original_jump_connections = model.jump_connections
        model.jump_connections = []  # Temporarily disable jump connections
        outputs_without_jumps = model(**inputs)

    # Perform inference with jump connections
    with torch.no_grad():
        model.jump_connections = original_jump_connections  # Re-enable jump connections
        outputs_with_jumps = model(**inputs)

    print("\nInference completed with and without jump connections.")
    print("Shape of logits:", outputs_with_jumps['logits'].shape)
    print("Number of hidden states:", len(outputs_with_jumps['hidden_states']))

    # Compute and print differences in final hidden states and logits
    final_state_diff = (outputs_with_jumps['last_hidden_state'] - outputs_without_jumps['last_hidden_state']).abs().mean().item()
    print(f"\nAverage absolute difference in final hidden state: {final_state_diff:.6f}")

    logits_diff = (outputs_with_jumps['logits'] - outputs_without_jumps['logits']).abs().mean().item()
    print(f"Average absolute difference in logits: {logits_diff:.6f}")

    # Print top 5 token predictions with and without jump connections
    top_tokens_without_jumps = torch.topk(outputs_without_jumps['logits'][0, -1], 5)
    top_tokens_with_jumps = torch.topk(outputs_with_jumps['logits'][0, -1], 5)

    print("\nTop 5 token predictions without jump connections:")
    for i in range(5):
        token = tokenizer.decode([top_tokens_without_jumps.indices[i]])
        prob = torch.softmax(outputs_without_jumps['logits'][0, -1], dim=0)[top_tokens_without_jumps.indices[i]].item()
        print(f"  {token}: {prob:.4f}")

    print("\nTop 5 token predictions with jump connections:")
    for i in range(5):
        token = tokenizer.decode([top_tokens_with_jumps.indices[i]])
        prob = torch.softmax(outputs_with_jumps['logits'][0, -1], dim=0)[top_tokens_with_jumps.indices[i]].item()
        print(f"  {token}: {prob:.4f}")

# Run inference and print results
run_inference_and_print()