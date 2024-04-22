from imageai.Detection import ObjectDetection
import os


# Uses ImageAI to detect objects in an image using a pretrained model.
def detect_objects(image):
    exec_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(exec_path, "Classes/yolov3.pt"))
    detector.loadModel()
    detected_objects = detector.detectObjectsFromImage(input_image=os.path.join(exec_path, image), output_image_path=os.path.join(exec_path, "Test Images/image_new.jpg"), minimum_percentage_probability=30)
    for eachObject in detected_objects:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        print("--------------------------------")

    return detected_objects


def goal():
    return (10, 10)
