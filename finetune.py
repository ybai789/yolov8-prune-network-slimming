import argparse
from ultralytics import YOLO

def train_model(opt):
    model = YOLO(opt.weights)
    results = model.train(data=opt.data, 
                          epochs=opt.epochs, 
                          imgsz=opt.imgsz, 
                          batch=opt.batch, 
                          workers=opt.workers, 
                          device=opt.device, 
                          sr=opt.sr, 
                          finetune=opt.finetune, 
                          name=opt.name)
    return results

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="./ultralytics/weights/pruned.pt", help='path to model weights')
    parser.add_argument('--data', type=str, default='./ultralytics/cfg/datasets/VOC.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--device', nargs='+', type=int, default=[0], help='device id(s) (e.g., 0 or 0 1 2 3)')
    parser.add_argument('--sr', type=float, default=0, help='sparse rate')
    # opt.finetune now defaulting to True unless overridden by specifying --no-finetune
    parser.add_argument('--no-finetune', action='store_false', dest='finetune', help='do not finetune the model')
    parser.add_argument('--name', type=str, default='train-finetune', help='save to project/name')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    results = train_model(opt)

