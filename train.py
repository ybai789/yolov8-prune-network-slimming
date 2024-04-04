import argparse
# Assuming ultralytics.YOLO exists in your setup or a similar API that you're using
from ultralytics import YOLO

def train_model(opt):
    # Load a pretrained model
    model = YOLO(opt.weights)  # Use the weights path provided by the command-line argument

    # Train the model
    results = model.train(data=opt.data, 
                        epochs=opt.epochs, 
                        imgsz=opt.imgsz, 
                        batch=opt.batch, 
                        workers=opt.workers, 
                        device=opt.device, 
                        sr=0, 
                        name=opt.name)
    return results

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./ultralytics/ultralytics/weights/yolov8s.pt', help='Path to the model weights file')
    parser.add_argument('--data', type=str, default='./ultralytics/ultralytics/cfg/datasets/VOC.yaml', help='Path to the dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', nargs='+', type=int, default=[0], help='Device ID or IDs for training (e.g., 0 or 0 1 for multiple GPUs)')
    parser.add_argument('--name', type=str, default='train', help='save to project/name')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    results = train_model(opt)

