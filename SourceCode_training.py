from ultralytics import YOLO
import cv2
import os

# Path to your dataset's YAML configuration file
DATA_YAML = r"C:\\Users\\VIT\\Desktop\\xyz\\annotated_dataset_roboflow\\data.yaml"




# Training the YOLOv8 Model
def train_yolov8():

    save_folder = r"C:\\Users\\VIT\\Desktop\\Model_saved"
    
    # Path to save the trained model weights
    MODEL_SAVE_PATH = os.path.join(save_folder, "best.pt")


    # Initialize a YOLOv8 model with pre-trained weights (Nano version)
    model = YOLO("yolov8n.pt")  # You can use "yolov8s.pt" or other variants

    # Train the model using the dataset
    model.train(
        data=DATA_YAML,  # Path to the YAML file 
        epochs=125,       # Number of epochs
        imgsz=640,       # Image size
        batch=16,        # Batch size
        device=0,  # Using the GPU (Nvidia Geforce RTX 3070-12GB VRAM) 
        name="Plant_Disease_Detection_Model"  # Experiment name
    )
    model.save(MODEL_SAVE_PATH)  
    print(f"Model trained and saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    
    train_yolov8()
    