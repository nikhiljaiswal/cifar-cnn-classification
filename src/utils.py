import torch
from torchsummary import summary
from typing import Tuple, Union
import io
from contextlib import redirect_stdout

def get_model_summary(model: torch.nn.Module, 
                     input_size: Union[Tuple, list] = (3, 32, 32),
                     batch_size: int = -1,
                     device: str = "cuda") -> str:
    """
    Get a string summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        input_size: Size of input tensor (channels, height, width)
        batch_size: Batch size for summary
        device: Device to run summary on
    
    Returns:
        str: Complete model summary
    """
    summary_str = io.StringIO()
    
    with redirect_stdout(summary_str):
        # Model architecture
        print("\nModel Architecture:")
        print("=" * 50)
        print(model)
        print("=" * 50)
        
        # Parameter summary
        print("\nParameter Summary:")
        print("=" * 50)
        summary(model, input_size=input_size, batch_size=batch_size, device=device)
        print("=" * 50)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("=" * 50)
    
    return summary_str.getvalue()
    
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

    accuracy = 100 * correct / processed
    print(f">>Epoch: {epoch}\nTrain Accuracy: {accuracy:.2f}%")
    return accuracy

def test_epoch(model, device, test_loader, criterion, scheduler):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    if scheduler:
        scheduler.step(test_loss)
        
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy 