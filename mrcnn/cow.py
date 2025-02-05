"""
Mask R-CNN, image segmentation model, for cow estrus detection
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/drive/Shareddrives/스마트축사_데이터_활용_대회/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("/content/drive/Shareddrives/스마트축사_데이터_활용_대회/Mask_RCNN/mrcnn")
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import imgaug
import imgaug.augmenters

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class CowConfig(Config):
    """Configuration for training on the cow dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cow"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1  # Background + 발정 + 비발정

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 64

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6


############################################################
#  Dataset
############################################################

class CowDataset(utils.Dataset):
    def load_cow(self, dataset_dir, subset):
        """Load a cow dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have two class to add.
        # class 0: background
        self.add_class("cow", 1, "anestrus")  # 비발정
        self.add_class("cow", 2, "estrus")  # 발정

        # Train or validation dataset?
        assert subset in ["train", "val"]

        # Load annotations (json file)
        # 경로
        json_file = json.load(open(os.path.join(dataset_dir, subset, subset + "_answer.json")))

        Images = json_file['images']  # 이미지
        annotations = pd.DataFrame(json_file['annotations'])  # 이미지 속 인스턴스

        # utils.Dataset에서 상속한 image_info 리스트에 이미지와 인스턴스의 정보를 저장
        for image in Images:
            # 이미지 파일 경로
            image_path = os.path.join(dataset_dir,
                                      subset + '/' + image['file_name'])

            # 이미지 파일의 높이/너비
            height, width = image['height'], image['width']

            # 이미지 속 인스턴스들의 테두리 좌표 및 class
            # 좌표값
            polygons = []
            for polygon in annotations.loc[annotations.image_id == image['id'],
                                           'segmentation']:
              polygon_resize = polygon.copy()
              for i, coord in enumerate(polygon):
                if i % 2 == 0 and coord >= width:
                  polygon_resize[i] = coord-1
                elif i % 2 == 1 and coord >= height:
                  polygon_resize[i] = coord-1
              polygons.append(polygon_resize)
            
            category_id = [x for x in annotations.loc[annotations.image_id == image['id'],
                                                      'category_id']]
            # image_info 리스트에 정보를 저장
            self.add_image(
                'cow',  # source
                image['id'],  # image_id
                image_path,
                width=width, height=height,
                polygons=polygons,
                category_id=category_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # mask.shape = [높이, 너비, 인스턴스의 개수]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        # Get indexes of pixels inside the polygon and set them to 1
        # skimage.draw.polygon(y point, x point)
        for i, p in enumerate(info['polygons']):
            rr, cc = skimage.draw.polygon(p[1::2], p[0::2])            
            mask[rr, cc, i] = 1

        # 인스턴스의 mask와 class 반환
        return mask.astype(np.bool), np.array(info['category_id'], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]

        ######## 수정 ##########
        if info["source"] == "cow":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, augmentation):
    """Train the model."""
    # Training dataset.
    dataset_train = CowDataset()
    dataset_train.load_cow(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CowDataset()
    dataset_val.load_cow(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                # learning_rate=config.LEARNING_RATE,
                learning_rate = float(args.learning_rate),
                epochs=int(args.epochs),
                layers=args.layers,
                augmentation=exec(args.augmentation))


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Cow.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--epochs', required=False,
                        default=30,
                        metavar="set the epoch",
                        help='Set the epoch')
    parser.add_argument('--learning_rate', required=False,
                        default=0.001,
                        metavar="set the learning_rate",
                        help='Set the learning_rate')
    parser.add_argument('--rpn_nms_threshold', required=False,
                        default=0.7,
                        metavar="set the rpn_nms_threshold",
                        help='Set the rpn_nms_threshold')
    parser.add_argument('--augmentation', required=False,
                        default=None,
                        metavar="augmentation",
                        help='augmentation : True or None')
    parser.add_argument('--layers', required=False,
                        default='heads',
                        metavar="layers",
                        help='layers : all, heads, 3+, 4+, 5+')
    
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Epochs: ", args.epochs)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("augmentation: ", args.augmentation)
    print("Layers: ", args.layers)

    # Configurations
    if args.command == "train":
        class TrainConfig(CowConfig):
            RPN_NMS_THRESHOLD = float(args.rpn_nms_threshold)
        config = TrainConfig()
    else:
        class InferenceConfig(CowConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, bool(args.augmentation))
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
