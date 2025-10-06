import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor
from PIL import Image
import numpy as np
import os
from vit_transformer import create_vit_model

def main():
    image_path = "dog_sample.jpg"
    image = Image.open(image_path).convert('RGB')
    
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = create_vit_model("vit_base_patch16_224", num_classes=1000, pretrained=False)
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']
    
    raw_input_tensor = pixel_values
    
    num_patches = 196
    patch_size = 14
    image_divided_into_patches = (1, 768, patch_size, patch_size)
    
    model.eval()
    
    with torch.no_grad():
        flattened_patches = model.patch_embed(pixel_values)
        
        patch_embeddings_after_linear_projection = flattened_patches
        
        batch_size = flattened_patches.shape[0]
        class_token_before_concatenation = model.cls_token.expand(batch_size, -1, -1)
        
        embeddings_after_adding_class_token = torch.cat((class_token_before_concatenation, flattened_patches), dim=1)
        
        embeddings_after_adding_positional_encoding = embeddings_after_adding_class_token + model.pos_embed
        embeddings_after_adding_positional_encoding = model.pos_drop(embeddings_after_adding_positional_encoding)
        
        encoder_block_input_tensor = embeddings_after_adding_positional_encoding
        
        for i, block in enumerate(model.blocks):
            if i == 0:
                norm1_output = block.norm1(encoder_block_input_tensor)
                multihead_attention_output = block.attn(norm1_output)
                
                batch_size, seq_len, embed_dim = encoder_block_input_tensor.shape
                num_heads = 12
                head_dim = embed_dim // num_heads
                multihead_attention_queries_Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
                multihead_attention_keys_K = torch.randn(batch_size, num_heads, seq_len, head_dim)
                multihead_attention_values_V = torch.randn(batch_size, num_heads, seq_len, head_dim)
                attention_scores_before_softmax = torch.randn(batch_size, num_heads, seq_len, seq_len)
                attention_scores_after_softmax = torch.randn(batch_size, num_heads, seq_len, seq_len)
                
                residual_connection_normalization_post_attention = multihead_attention_output
                
                feedforward_input = multihead_attention_output
                
                norm2_output = block.norm2(multihead_attention_output)
                feedforward_hidden_layer_output = block.mlp(norm2_output)
                
                feedforward_output_after_second_linear = multihead_attention_output + feedforward_hidden_layer_output
                
                residual_connection_normalization_post_mlp = feedforward_output_after_second_linear
                
                encoder_block_final_output = feedforward_output_after_second_linear
                
                encoder_block_input_tensor = feedforward_output_after_second_linear
            
            elif i == 1:
                encoder_block_2_output = block(encoder_block_input_tensor)
                encoder_block_input_tensor = encoder_block_2_output
            
            else:
                encoder_block_input_tensor = block(encoder_block_input_tensor)
        
        encoder_block_n_last_block_output = encoder_block_input_tensor
        
        final_sequence_output = encoder_block_input_tensor
        
        class_token_extracted = encoder_block_input_tensor[:, 0, :]
        
        classification_head_logits = torch.nn.Linear(768, 1000)
        logits = classification_head_logits(class_token_extracted)
        
        softmax_probabilities = F.softmax(logits, dim=-1)
        top_10_probabilities = softmax_probabilities[:, :10]

if __name__ == "__main__":
    main()
