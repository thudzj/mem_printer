import cPickle as pickle
import os.path as osp

import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn.datasets.base import Bunch
import theano
import plyvel


def labelcolormap(N=256):
    cmap = np.zeros((N, 3))
    for i in xrange(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in xrange(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

class SegmentationClassDataset(Bunch):

    target_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    def __init__(self, db_path = "./db", pascald = "./voc12/VOCdevkit/VOC2012"):
        super(self.__class__, self).__init__()
        self.db = plyvel.DB(db_path, create_if_missing=True)
        self.pascal_dir = pascald
        self.train_list = [id_.strip() for id_ in open(osp.join(self.pascal_dir, 'ImageSets/Segmentation/{0}.txt'.format("train")))]
        self.trainval_list = [id_.strip() for id_ in open(osp.join(self.pascal_dir, 'ImageSets/Segmentation/{0}.txt'.format("trainval")))]
        self.val_list = [id_.strip() for id_ in open(osp.join(self.pascal_dir, 'ImageSets/Segmentation/{0}.txt'.format("val")))]
        self.train_index = 0
        self.trainval_index = 0
        self.val_index = 0
        cmap = labelcolormap(len(self.target_names))
        self.cmap = (cmap * 255).astype(np.uint8)


    def _crop(self, img, label):
        assert img.shape[1] == label.shape[0]
        assert img.shape[2] == label.shape[1]
        #print img.shape
        h = np.random.randint(img.shape[1] - 224 + 1, size=1)[0]
        w = np.random.randint(img.shape[2] - 224 + 1, size=1)[0]
        img = img[:, h:h+224, w:w+224]
        label = label[h:h+224, w:w+224]
        return img, label

    def _load_datum(self, id):
        # check cache
        datum = self.db.get(str(id))
        if datum is not None:
            img, label = pickle.loads(datum)
            return self._crop(img, label)
        # there is no cache
        img_file = osp.join(self.pascal_dir, 'JPEGImages', id + '.jpg')
        img = imread(img_file, mode='RGB')
        if img.shape[0] < 224 or img.shape[1] < 224:
            print "------>",img.shape
            img = imresize(img, [max(224, img.shape[0]), max(224, img.shape[1])])
        img = self.img_to_datum(img)

        label_rgb_file = osp.join(self.pascal_dir, 'SegmentationClass', id + '.png')
        label_rgb = imread(label_rgb_file, mode='RGB')
        if label_rgb.shape[0] < 224 or label_rgb.shape[1] < 224:
            label_rgb = imresize(label_rgb, [max(224, label_rgb.shape[0]), max(224, label_rgb.shape[1])], interp='nearest')
        label = self._label_rgb_to_32sc1(label_rgb)
        
        datum = (img, label)
        # save cache
        self.db.put(str(id), pickle.dumps(datum))
        
        return self._crop(img, label)

    def _label_rgb_to_32sc1(self, label_rgb):
        assert label_rgb.dtype == np.uint8
        label = np.zeros(label_rgb.shape[:2], dtype=np.int32)
        label.fill(-1)
        
        for l, rgb in enumerate(self.cmap):
            mask = np.all(label_rgb == rgb, axis=-1)
            label[mask] = l
        return label

    def _label_32sc1_to_rgb(self, label):
        #print label.shape
        label_rgb = np.zeros([label.shape[0], label.shape[1], 3], np.uint8)

        for l, rgb in enumerate(self.cmap):
            label_rgb[label == l] = rgb
        return label_rgb

    def img_to_datum(self,img):
        datum = img.astype(np.float32)
        datum = datum[:, :, ::-1]  # RGB -> BGR
        datum -= np.array((104.00698793, 116.66876762, 122.67891434))
        datum = datum.transpose((2, 0, 1))
        return datum

    def random_ins(self):
        inx = np.random.randint(len(self.val_list), size=1)[0]
        return self._load_datum(self.val_list[inx]), self.val_list[inx]

    def save_label(self, label, name):
        label_rgb_file = osp.join(self.pascal_dir, 'gen', name + '.png')
        i = self._label_32sc1_to_rgb(label)
        #print i.shape
        imsave(label_rgb_file, i)

    def next_train_batch(self, batch_size):
        """Generate next batch whose size is the specified batch_size."""
        start = self.train_index
        end = self.train_index + batch_size
        if end > len(self.train_list):
            np.random.shuffle(self.train_list)
            start = 0
            end = batch_size
        self.train_index = end
        batch_data = []
        batch_label = []
        for id in self.train_list[start:end]:
            img, label = self._load_datum(id)
            batch_data.append(img)
            batch_label.append(label)
        return np.asarray(batch_data, dtype=theano.config.floatX), np.asarray(batch_label, dtype=theano.config.floatX)

    def next_trainval_batch(self, batch_size):
        """Generate next batch whose size is the specified batch_size."""
        start = self.trainval_index
        end = self.trainval_index + batch_size
        if end > len(self.trainval_list):
            np.random.shuffle(self.trainval_list)
            start = 0
            end = batch_size
        self.trainval_index = end
        batch_data = []
        batch_label = []
        for id in self.trainval_list[start:end]:
            img, label = self._load_datum(id)
            batch_data.append(img)
            batch_label.append(label)
        return np.asarray(batch_data, dtype=theano.config.floatX), np.asarray(batch_label, dtype=theano.config.floatX)

    def next_val_batch(self, batch_size):
        """Generate next batch whose size is the specified batch_size."""
        start = self.val_index
        end = self.val_index + batch_size
        if end > len(self.val_list):
            np.random.shuffle(self.val_list)
            start = 0
            end = batch_size
        self.val_index = end
        batch_data = []
        batch_label = []
        for id in self.val_list[start:end]:
            img, label = self._load_datum(id)
            batch_data.append(img)
            batch_label.append(label)
        return np.asarray(batch_data, dtype=theano.config.floatX), np.asarray(batch_label, dtype=theano.config.floatX)
