import caffe
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

caffe.set_mode_cpu()

# caffe.set_device(0)
# caffe.set_mode_gpu()

root_dir = "/home/tim/datasets/cifar10/samples/"

# load the class labels from disk
rows = open(root_dir + 'synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

model_def = "cifar10_googlenet_deploy.prototxt"
model_weights = 'snapshot/cifar10_googlenet_iter_120000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

from imutils import paths
imagePaths = sorted(list(paths.list_images(root_dir)))

for filename in imagePaths:
    # transform it and copy it into the net
    # [0, 1]
    image = caffe.io.load_image(filename=filename, color=True)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # (32,32,3) ==> (3,32,32)
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension

    """
    compute_image_mean.cpp:119] mean_value channel [0]: 125.284 
    I1121 11:03:15.626080 1126 compute_image_mean.cpp:119] mean_value channel [1]: 122.947 
    I1121 11:03:15.626085 1126 compute_image_mean.cpp:119] mean_value channel [2]: 113.86
    """

    mu = np.array([125.284, 122.947, 113.86])
    print('mean-subtracted values:', zip('BGR', mu))


    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    bt = cv2.getTickCount()
    net.blobs['data'].data[...] = transformer.preprocess('data', image)

    # perform classification
    net.forward()

    et = cv2.getTickCount()

    print("--inference time: {0} ms--".format((et - bt) * 1000.0 / cv2.getTickFrequency()))

    # obtain the output probabilities
    output_prob = net.blobs['prob'].data[0]

    print(type(output_prob))

    classId = np.argmax(output_prob)
    confidence = output_prob[classId]
    '''
    pred_label=airplane, confidence=1.0000
    pred_label=ship, confidence=0.7825
    pred_label=cat, confidence=0.5985
    pred_label=cat, confidence=0.9973
    pred_label=deer, confidence=0.6061
    pred_label=dog, confidence=1.0000
    pred_label=frog, confidence=0.9996
    pred_label=horse, confidence=1.0000
    pred_label=ship, confidence=1.0000
    pred_label=truck, confidence=0.9938
    '''
    label = 'pred_label=%s, confidence=%.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
    print(label)

    # plt.imshow(image)
    #
    # plt.show()
    cv2.imshow("img", image)
    cv2.waitKey(0)