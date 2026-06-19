import torch
import tiktoken
from src.config import CONFIG_124M
from src.dataset import create_dataloader_v1
from src.model import BaseModel
from src.utils import calc_loss_batch, evaluate_model, generate_and_print_sample

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() 
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    # 1. Setup device
    if torch.cuda.is_available():  
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        device = torch.device("mps") if (major, minor) >= (2, 9) else torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")

    # 2. Load Text Data
    with open("data/the-verdict.txt", "r") as f: # Assumes text is moved to a data/ folder
        text_data = f.read()

    # 3. Create Data Splits & Loaders
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data, batch_size=2, max_length=CONFIG_124M["context_length"],
        stride=CONFIG_124M["context_length"], drop_last=True, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data, batch_size=2, max_length=CONFIG_124M["context_length"],
        stride=CONFIG_124M["context_length"], drop_last=False, shuffle=False, num_workers=0
    )

    # 4. Initialize Model & Optimizer
    tokenizer = tiktoken.encoding_for_model("gpt2")
    model = BaseModel(CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # 5. Train
    print("Starting training...")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=10, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # 6. Save State
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        "model_and_optimizer.pth"
    )
    print("Model saved to model_and_optimizer.pth")