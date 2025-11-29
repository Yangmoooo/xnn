import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import FashionMNISTDataset
from model import NeuralNetwork

# --- Hyperparameters ---
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 50
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
TRAIN_SPLIT = 0.8
TRAIN_CSV_PATH = "../data/fashion-mnist/fashion-mnist_train.csv"
TEST_CSV_PATH = "../data/fashion-mnist/fashion-mnist_test.csv"


def evaluate(model, data_loader, device):
    """Evaluates the model on a given dataset."""
    model.eval()  # Set the model to evaluation mode (disables dropout)
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy, correct_predictions


def main():
    """Main function to run training and evaluation."""
    # --- Device Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data Loading ---
    full_train_dataset = FashionMNISTDataset(TRAIN_CSV_PATH, device=device)
    test_dataset = FashionMNISTDataset(TEST_CSV_PATH, device=device)

    # Split training data into training and validation sets
    train_size = int(TRAIN_SPLIT * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_test = len(test_dataset)
    input_size = full_train_dataset.images.shape[1]

    print(f"Training on {n_train} samples, validating on {n_val} samples.")

    # --- Network Initialization ---
    print("Initializing network...")
    model = NeuralNetwork(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()  # Set the model to training mode (enables dropout)
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()  # Clear gradients from previous iteration
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            total_loss += loss.item() * images.size(0)

        # --- Validation ---
        val_accuracy, _ = evaluate(model, val_loader, device)
        avg_loss = total_loss / n_train
        end_time = time.time()

        print(
            f"Epoch {epoch + 1:2d}, Accuracy: {val_accuracy * 100:6.2f}%, "
            f"Avg Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f} seconds"
        )

    # --- Final Evaluation on Test Set ---
    print("Starting final evaluation on the test set...")
    test_accuracy, n_correct = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}% ({n_correct}/{n_test})")


if __name__ == "__main__":
    main()
