import os
import argparse
from model import SegmentCarNumber

model = SegmentCarNumber(encoder_path="encoder_weights.pth", unet_path="unet_model.pth")

files = os.listdir('.')
usls = ['main.py', 'model.py', 'encoder_weights.pth', 'unet_model.pth', 'requirements.txt']
for f in usls:
    if f in files:
        files.remove(f)
if len(files) == 1:
    # тут запускаем с помощью docker
    model.predict_dir('images', 'images/result')
else:
    # тут с помощью python
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Директория с входными данными")
    parser.add_argument("output_folder", type=str, help="Директория с выходными данными")
    args = parser.parse_args()
    model.predict_dir(args.input_folder, args.output_folder)
