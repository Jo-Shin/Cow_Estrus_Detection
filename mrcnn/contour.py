import numpy as np
from skimage.measure import find_contours
import itertools as it

def get_contours(image, boxes, masks, class_ids, class_names,
                      scores=None,
                      colors=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    object_contour = []
    for i in range(N):

        # Label: caption
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        if label == 'estrus':
            categoryId = 2
        elif label == 'anestrus':
            categoryId = 1

        # caption = "{} {:.3f}".format(label, score) if score else label


        # Mask
        mask = masks[:, :, i]

        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        # print(contours[0].shape)
        for index, verts in enumerate(contours):
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            x = verts[:, 0]
            y = verts[:, 1]
            segmentation_lst = list(it.chain(*zip(x, y)))
        object_contour.append({'image_id': "", 'segmentation': segmentation_lst, 'category_id': categoryId, "conf": round(score, 3)})

    return object_contour