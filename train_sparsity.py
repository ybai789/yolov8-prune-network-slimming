import argparse
from ultralytics import YOLO

def train_model(opt):
    # Load a model using the path specified in opt.weights
    model = YOLO(opt.weights)

    # Train the model with parameters specified in the opt argument
    results = model.train(data=opt.data, 
                          epochs=opt.epochs, 
                          imgsz=opt.imgsz, 
                          batch=opt.batch, 
                          workers=opt.workers, 
                          device=opt.device, 
                          sr=opt.sr,
                          name=opt.name)
    return results

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./ultralytics/ultralytics/weights/yolov8s.pt', help='path to model weights')
    parser.add_argument('--data', type=str, default='./ultralytics/ultralytics/cfg/datasets/VOC-ball.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='input image size')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--device', nargs='+', type=int, default=[0], help='device id(s) for training, e.g., 0 or 0 1 2 3')
    parser.add_argument('--sr', type=float, default=0.005, help='L1 regularization penalty coefficient (sparsity regularization)')
    parser.add_argument('--name', type=str, default='train-sparse', help='save to project/name')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    results = train_model(opt)
