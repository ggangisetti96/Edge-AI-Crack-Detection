import pygame
import pygame.camera
import os
import pathlib
import time
import sys
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def normalize_img(image):
    params = common.input_details(interpreter, 'quantization_parameters')
    scale = params['scales']
    zero_point = params['zero_points']
    mean = 128.0
    std = 128.0
    if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
        # Input data does not require preprocessing.
        normalized_image = np.asarray(image)
    else:
        # Input data requires preprocessing
        normalized_image = (np.asarray(image) - mean) / (std * scale) + zero_point
        np.clip(normalized_image, 0, 255, out=normalized_image)
        normalized_image = normalized_image.astype(np.uint8)
    
    return normalize_img

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'CNN_6Layers_tflite_edgetpu.tflite')
label_file = os.path.join(script_dir, 'labels.txt')

interpreter = make_interpreter(model_file)
labels = read_label_file(label_file)
interpreter.allocate_tensors()

req_normalization = False
# Model must be uint8 quantized
if common.input_details(interpreter, 'dtype') == np.uint8:
    print('Only support uint8 input type.')
    req_normalization = True

size = common.input_size(interpreter)

pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Arial', 20)

# Initialize the camera
pygame.camera.init()
camlist = pygame.camera.list_cameras()

if camlist:
    cam = pygame.camera.Camera(camlist[1], (640, 480))
    cam.start()
else:
    print("No camera on current device")

try:
    display = pygame.display.set_mode((640, 480))
except pygame.error as e:
    sys.stderr.write("\nERROR: Unable to open a display window. Make sure a monitor is attached and that "
            "the DISPLAY environment variable is set. Example: \n"
            ">export DISPLAY=\":0\" \n")
    raise e

red = pygame.Color(255, 0, 0)

try:
    clock = pygame.time.Clock()
    last_time = time.monotonic()


    while True:

        # Set the frame rate to 30 FPS
        clock.tick(30)

        # Capture an image from the camera
        mysurface = cam.get_image()

        # Convert the image to a PIL Image object
        image = Image.fromarray(pygame.surfarray.array3d(mysurface))

        # Resize the image to the input size of the model
        image = image.resize(size, Image.ANTIALIAS)

        # Check if the input data requires normalization and quantization preprocessing
        if req_normalization:
            image = normalize_img(image)
        else:
            image = np.asarray(image)/255

        # Set the preprocessed image as input to the model
        common.set_input(interpreter, image)

        # Run inference
        print('----INFERENCE TIME----')
        print('Note: The first inference on Edge TPU is slow because it includes',
            'loading the model into Edge TPU memory.')
        start_time = time.monotonic()
        interpreter.invoke()
        stop_time = time.monotonic()
        classes = classify.get_classes(interpreter, 1, 0.0)
        inference_ms = (stop_time - start_time)*1000.0
        inference_ms = inference_ms
        fps_ms = 1.0 / (stop_time - last_time)
        last_time = stop_time
        annotate_text = 'Inference: {:5.2f}ms FPS: {:3.1f}'.format(inference_ms, fps_ms)
        for i in range(len(classes)):
            c=classes[i]
            label = '{:.0f}% {}'.format(100*c.score, labels.get(c.id, c.id))
            text = font.render(label, True, red)
            # Print the results
            print(label, " ", end="")
            mysurface.blit(text, (0, 20))
        text = font.render(annotate_text, True, red)
        print(annotate_text)
        mysurface.blit(text, (0, 0))
        display.blit(mysurface, (0, 0))
        pygame.display.flip()

finally:
    # Release the camera
    cam.stop()
