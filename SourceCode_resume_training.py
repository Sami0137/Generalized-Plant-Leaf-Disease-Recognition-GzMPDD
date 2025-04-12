from ultralytics import YOLO
import cv2
import os


# Path to your dataset's YAML configuration file
DATA_YAML = r"C:\\Users\\VIT\\Desktop\\xyz\\annotated_dataset_roboflow\\data.yaml"  # Dataset YAML path

# Path to the folder where the model from the previous run is saved
PREVIOUS_WEIGHTS_PATH = r"C:\\Users\\VIT\\Desktop\\Model_saved\\best.pt"  # Update with the actual path of your previously trained model

# Path to save the updated weights and logs for this training session
SAVE_FOLDER = r"C:\\Users\\VIT\\Desktop\\Resumed_Model_saved" # Update this if needed

# Step 2: Resume Training YOLOv8 Model
def resume_training():
    # Initialize the model with the previously trained weights
    model = YOLO(PREVIOUS_WEIGHTS_PATH)

    # Resume training from where it left off
    model.train(
        data=DATA_YAML,    # Path to the YAML file
        epochs=121,        # Total epochs (new epochs + previous)
        imgsz=640,         # Image size
        batch=16,          # Batch size
        device=0,          # GPU device (or "cpu")
        name="Plant_Disease_Detection_Model_resumed"  # New experiment name
    )
    
    # Save the updated weights
    updated_model_path = os.path.join(SAVE_FOLDER, "best_rzmd.pt")
    model.save(updated_model_path)
    print(f"Model continued training and saved at: {updated_model_path}")

if __name__ == "__main__":
    # Run the resumed training
    resume_training()


