import cv2
import cv2 as cv
import numpy as np

root_dir = "/home/tim/datasets/cifar10/samples/"

# load the class labels from disk
rows = open(root_dir + 'synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# Load a network
# model = 'snapshot/cifar10_googlenet_iter_20000.caffemodel'
# config = 'cifar10_googlenet_deploy.prototxt'

model_def = "/home/tim/myWorkSpace/cifar10_classification/cifar10_resnet/cifar10_resnet56_deploy.prototxt"
model_weights = '/home/tim/myWorkSpace/cifar10_classification/cifar10_resnet/snapshot/cifar10_resnet56_iter_64000.caffemodel'

framework = 'caffe'

# net = cv.dnn.readNet(model, config, framework)
net = cv.dnn.readNet(model_weights, model_def, framework)

# backend = cv.dnn.DNN_BACKEND_DEFAULT
# target = cv.dnn.DNN_TARGET_CPU
# net.setPreferableBackend(backend)
# net.setPreferableTarget(target)


# grab the paths to the input images
from imutils import paths
imagePaths = sorted(list(paths.list_images(root_dir)))

for imagePath in imagePaths:
    frame = cv2.imread(imagePath, cv2.IMREAD_COLOR)

    # Create a 4D blob from a frame.
    scale = 1.0
    width = 32
    height = 32
    mean = [125.284, 122.947, 113.86]

    blob = cv.dnn.blobFromImage(image=frame, scalefactor=scale,
                                size=(width, height), mean=mean, swapRB=False)
    # Run a model
    net.setInput(blob)
    out = net.forward()

    # print("out: ", out)
    # Get a class with a highest score.
    out = out.flatten()
    classId = np.argmax(out)
    confidence = out[classId]

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    print(label)

    # Print predicted class.
    label = 'pred_label=%s, confidence=%.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
    # cv.putText(frame, label, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    print(label)

    cv2.imshow('img', frame)
    key = cv2.waitKey(0)