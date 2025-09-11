from dataset_loader import DatasetLoader
from model import create_cnn_model

# Load dummy dataset
loader = DatasetLoader("dummy_dataset")
X, y = loader.load_data()

# Create CNN model
model = create_cnn_model()

# Dummy training (just 1 epoch to show)
print("Starting dummy training...")
model.fit(X.reshape(-1,224,224,1), y, epochs=1, batch_size=8)
print("Training complete!")
