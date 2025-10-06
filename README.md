# Vision Transformer (ViT) Debugging Assignment

## Project Overview

This project implements and debugs a Vision Transformer (ViT) model for educational purposes. The assignment focuses on understanding the internal workings of transformer architectures applied to computer vision tasks through step-by-step debugging and analysis.

## Project Structure

```
Assignment_2/
├── vit_transformer.py      # Custom ViT implementation
├── vit_debugging.py        # Debugging script with 26 snapshots
├── dog_sample.jpg          # Test image (Golden retriever)
├── Assignment_2_Report_Template.txt  # Report template
├── BREAKPOINT_GUIDE.md     # Debugging guide
├── DEBUGGING_GUIDE.md      # Detailed debugging instructions
└── README.md              # This file
```

## Requirements

### Environment Setup
- **Python**: 3.10.18 (WSL Ubuntu)
- **IDE**: PyCharm with WSL integration
- **Virtual Environment**: `.venv` (excluded from repository)

### Dependencies
```bash
pip install torch torchvision transformers pillow numpy
```

### Model Configuration
- **Model**: `google/vit-base-patch16-224`
- **Image Size**: 224×224 pixels
- **Patch Size**: 16×16 pixels
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Encoder Blocks**: 12
- **Classification Classes**: 1000 (ImageNet)

## Files Description

### `vit_transformer.py`
Custom implementation of Vision Transformer architecture including:
- `PatchEmbedding`: Converts images to patch embeddings
- `MultiHeadSelfAttention`: Multi-head attention mechanism
- `MLP`: Feed-forward network
- `TransformerBlock`: Complete transformer encoder block
- `VisionTransformer`: Full ViT model implementation

### `vit_debugging.py`
Debugging script that captures 26 snapshots during ViT forward pass:
- Input preprocessing and patch embedding
- Class token and positional encoding
- Multi-head attention mechanism
- Feed-forward network processing
- Final classification output

## Debugging Process

The debugging process involves setting breakpoints at 26 specific locations to capture tensor shapes and values at each processing stage:

1. **Input Processing** (Snapshots 1-2)
2. **Patch Embedding** (Snapshots 3-4)
3. **Token Addition** (Snapshots 5-7)
4. **Encoder Processing** (Snapshots 8-22)
5. **Final Output** (Snapshots 23-26)

## Usage

### Running the Debugging Script
```bash
python vit_debugging.py
```



## Assignment Objectives

1. **Environment Setup**: Configure PyCharm with WSL integration
2. **Model Understanding**: Analyze ViT architecture components
3. **Debugging Skills**: Use PyCharm debugger to inspect tensor operations
4. **Shape Tracking**: Document tensor shapes throughout the forward pass
5. **Documentation**: Create comprehensive debugging report with screenshots
