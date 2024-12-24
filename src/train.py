import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import logging
from datetime import datetime

from model import CIFAR10Net
from dataset import get_dataloaders
from utils import get_model_summary, train_epoch, test_epoch
from config import Config

def setup_logging():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_data_availability():
    """Check if CIFAR-10 dataset is already downloaded"""
    train_path = os.path.join('data', 'cifar-10-batches-py')
    return os.path.exists(train_path)

def save_checkpoint(state, filename):
    """Save model checkpoint"""
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(state, filename)

def main():
    logger = setup_logging()
    logger.info("Starting training process")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model initialization
    logger.info("Initializing model")
    model = CIFAR10Net(
        num_classes=Config.NUM_CLASSES, 
        dropout_value=Config.DROPOUT_VALUE
    ).to(device)
    
    # Get and log model summary
    logger.info("Model Summary:")
    model_summary = get_model_summary(model, input_size=(3, 32, 32), device=str(device))
    logger.info("\n" + model_summary)

    # Check data availability and get dataloaders
    data_exists = check_data_availability()
    logger.info("CIFAR-10 dataset status: " + ("Found" if data_exists else "Not found, downloading"))
    
    train_loader, test_loader = get_dataloaders(
        Config.MEAN, Config.STD, Config.BATCH_SIZE
    )
    logger.info(f"Dataloaders initialized with batch size: {Config.BATCH_SIZE}")

    # Criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        momentum=Config.MOMENTUM, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, 
        patience=3, min_lr=0.0001, verbose=True
    )

    logger.info(f"Training config - LR: {Config.LEARNING_RATE}, "
                f"Momentum: {Config.MOMENTUM}, "
                f"Weight Decay: {Config.WEIGHT_DECAY}")

    # Training loop
    best_accuracy = 0
    logger.info(f"Starting training for {Config.EPOCHS} epochs")

    try:
        for epoch in range(1, Config.EPOCHS + 1):
            logger.info(f"\nEpoch {epoch}/{Config.EPOCHS}")
            
            # Training phase
            train_accuracy = train_epoch(
                model, device, train_loader, optimizer, criterion, epoch
            )
            logger.info(f"Training Accuracy: {train_accuracy:.2f}%")
            
            # Testing phase
            test_accuracy = test_epoch(
                model, device, test_loader, criterion, scheduler
            )
            logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
            
            # Save checkpoint if best accuracy
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_accuracy,
                }
                save_checkpoint(
                    checkpoint,
                    f'checkpoints/model_epoch{epoch}_acc{test_accuracy:.2f}.pth'
                )
                logger.info(f"New Best Test Accuracy: {best_accuracy:.2f}%")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

    logger.info("Training completed!")
    logger.info(f"Best Test Accuracy achieved: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main() 