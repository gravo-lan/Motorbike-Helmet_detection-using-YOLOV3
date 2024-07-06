import cv2
import numpy as np
from utils import label_map_util
import tensorflow as tf
from utils import visualization_utils as vis_util
import pytesseract

class ObjectDetector:
    def __init__(self, helmet_inference_path, frozen_graph_path, labelmap, number_of_classes):
        self.helmet_inference_path = helmet_inference_path
        self.frozen_graph_path = frozen_graph_path
        self.labelmap = labelmap
        self.number_of_classes = number_of_classes
        self.detection_graph = tf.Graph()

    def detection(self, input):

        # Name of the directory containing the object detection module we're using
        TRAINED_MODEL_DIR = self.helmet_inference_path

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = TRAINED_MODEL_DIR + self.frozen_graph_path
        print(PATH_TO_CKPT)
        # Path to label map file
        PATH_TO_LABELS = TRAINED_MODEL_DIR + self.labelmap

        # Number of classes the object detector can identify
        NUM_CLASSES = self.number_of_classes

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.

        print("> ====== Loading frozen graph into memory")

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=self.detection_graph)
            print(">  ====== Inference graph loaded.")

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        img = input

        # Load the Tensorflow model into memory.
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        sess = tf.Session(graph=self.detection_graph)
        image = cv2.imread(img)
        image = cv2.resize(image, (1080, 1080))
        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        return category_index,image, boxes, scores, classes, num


def visualise(frame, boxes, scores, classes, category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=6,
            min_score_thresh=0.70)


def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer


def main():
    # Initialize video capture from webcam (change path if using a video file)
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    detector = ObjectDetector('frozen_graphs', '/frozen_inference_graph_helmet.pb','/labelmap_helmet.pbtxt', 2)
    writer = create_video_writer(cap, "output.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream

        # Perform object detection on the current frame
        category_index_helmet, _, boxes_helmet, scores_helmet, classes_helmet = detector.detection(frame)

        # Visualize detected objects on the frame
        visualise(frame, boxes_helmet, scores_helmet, classes_helmet, category_index_helmet)

        # Extract license plate text using Tesseract OCR
        for box, score, cls in zip(boxes_helmet[0], scores_helmet[0], classes_helmet[0]):
            if score > 0.7:  # Adjust the confidence threshold as needed
                ymin, xmin, ymax, xmax = box
                plate_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                plate_text = pytesseract.image_to_string(plate_image, lang='eng', config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                print(f"Detected license plate: {plate_text}")

        # Display the frame
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow('Object detector', frame)
        writer.write(frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    writer.release
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()