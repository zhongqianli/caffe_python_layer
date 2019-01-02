import caffe
import json
import cv2
import numpy as np
import random

# 4 pixel pad, random crop
# img: 64x3x32x32
def zeropadding_and_crop(data):
    # # cifar10
    # # padding_img = np.pad(img, ((4, 4), (4, 4), (4, 4)), "constant", padder=0)
    padding_img = np.zeros((np.shape(data)[0], 3, 40, 40), dtype=np.float32)
    padding_img[..., 4:36, 4:36] = data[...]
    # #
    # cv2.imshow("pad", data[0][0])
    row_rand_num = random.randrange(9)
    col_rand_num = random.randrange(9)
    croped_img = padding_img[..., row_rand_num : row_rand_num + 32, col_rand_num : col_rand_num + 32]

    return croped_img

class DataAugmentationLayer(caffe.Layer):
    def setup(self, bottom, top):
        # print("setup")
        pass
    def reshape(self, bottom, top):
        # print("reshape")
        top[0].reshape(*bottom[0].data.shape)
        pass

    def forward(self, bottom, top):
        # print("forward")
        # blob: K X C x H X W
        # mnist: 64 x 1 x 28 x 28
        # cifar10: 64 x 3 x 32 x 32
        # a batch of images, K X C x H X W
        # top[0].data[...] = bottom[0].data
        #
        # print(np.shape(bottom[0].data))

        # # C channels
        # cimg = bottom[0].data[0]
        #
        # # one channel
        # img = bottom[0].data[0][0]
        #
        # for cimg in bottom[0].data:
        #     print("cimg:{0}".format(np.shape(cimg)))
        #     for img in cimg:
        #         print("img:{0}".format(np.shape(img)))
        #         cv2.imshow("img", img)
        #         cv2.waitKey(0)

        top[0].data[...] = zeropadding_and_crop(bottom[0].data)
        pass

    def backward(self, top, propagate_down, bottom):
        # print("backward")
        # bottom[0].diff[...] = top[0].diff
        pass

# opencv: rows x cols x channel
def zeropadding(img, pad_h=1, pad_w=1):
    print(np.shape(img))
    rows, cols, ch = np.shape(img)
    # # cifar10
    pad_img = np.zeros((rows + 2*pad_h, cols + 2*pad_w, ch), dtype=np.uint8)
    pad_img[pad_h : rows + pad_h, pad_w : cols + pad_w] = img[...]
    return pad_img

if __name__ == "__main__":
    img = cv2.imread("/home/tim/datasets/cifar10/samples/0.jpg", 1)
    img = zeropadding(img, 32, 64)
    cv2.imshow("img", img)
    cv2.waitKey(0)