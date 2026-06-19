# Autoregressive Language model adopting Decoder only transformer architecture 

An implementation of a 124M parameter  autoregressive language model with the decoder-only transformer architecture built entirely from scratch using PyTorch trained on the custom text file 

## Repository Structure

``` text
Native-language_model-125M/
│
├── data/
│   └── the-verdict.txt          # Pretraining source text
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Hyperparameters & model configurations
│   ├── dataset.py               # GPTDatasetV1 and create_dataloader
│   ├── modules.py               # LayerNorm, GELU, FeedForward, Attention components
│   ├── model.py                 # Core GPTModel assembly
│   └── utils.py                 # Text generation & helper functions
│
├── train.py                     # CLI script to execute the pretraining loop
├── GPT_based_LLM.ipynb          # Refactored notebook for interactive experimentation
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies    
              
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model locally: `python train.py`
3. Try interactive text generation: Open `GPT_based_LLM.ipynb`

*Note: You can train the model from scratch locally using `train.py` to generate the `model_and_optimizer.pth` checkpoint file.*