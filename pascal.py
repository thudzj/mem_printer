import cPickle as pickle
import os.path as osp

import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn.datasets.base import Bunch
import tensorflow as tf
import plyvel, math
import arrow

def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

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

    def __init__(self, db_path = "./db", pascald = "../deep_memory/voc12/VOCdevkit/VOC2012"):
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
        self.hist_label = [0] * 21
        self.hist_pre = [0] * 21
        self.hist_iu = [0] * 21
        self.test_img = None
        self.test_label = None
        self.mem_cnt = 0

    def reset_test(self):
        self.val_index = 0
        self.hist_label = [0] * 21
        self.hist_pre = [0] * 21
        self.hist_iu = [0] * 21

    def shuffle_test(self):
        np.random.shuffle(self.val_list)

    def test_size(self):
        self.test_img, self.test_label = self._load_datum(self.val_list[self.val_index])
        self.val_index += 1
        shape = self.test_img.shape[:2]
        return [int(math.ceil(shape[0] / 224.0)), int(math.ceil(shape[1] / 224.0))]

    def test_crop(self, i, j):
        shape = self.test_img.shape[:2]
        h_start = i * 224
        w_start = j * 224
        h_end = min(h_start + 224, shape[0])
        w_end = min(w_start + 224, shape[1])
        img_crop = self.test_img[h_start:h_end, w_start:w_end,:]
        #label_crop = self.test_label[h_start:h_end, w_start:w_end,:]
        pad0 = (0,0)
        pad1 = (0,0)
        if img_crop.shape[0] < 224:
            pad0 = (0, 224 - img_crop.shape[0])
        if img_crop.shape[1] < 224:
            pad1 = (0, 224 - img_crop.shape[1])
        pad2 = (0,0)
        img_crop = np.pad(img_crop, pad_width=(pad0, pad1, pad2), mode='constant', constant_values=0)
        #label_crop = np.pad(label_crop, pad_width=(pad0, pad1), mode='constant', constant_values=0)
        return img_crop.reshape([1, 224, 224, 3])#, label_crop.reshape([1, 224, 224])

    def cal_crop_hist(self, pre_label, i, j):
        shape = self.test_img.shape[:2]
        h_start = i * 224
        w_start = j * 224
        h_len = min(224, shape[0] - h_start)
        w_len = min(224, shape[1] - w_start)
        label_crop = self.test_label[h_start:(h_start+h_len), w_start:(w_start+w_len)]
        pre_label_crop = pre_label[:h_len, :w_len]
        for i in range(21):
            label_crop_mask = (label_crop == i)
            pre_label_crop_mask = ((pre_label_crop == i) & (label_crop != -1))
            self.hist_label[i] += np.sum(label_crop_mask)
            self.hist_pre[i] += np.sum(pre_label_crop_mask)
            self.hist_iu[i] += np.sum(label_crop_mask & pre_label_crop_mask)

    def cal_iou(self):
        res = [0] * 21
        for i in range(21):
            if self.hist_label[i] + self.hist_pre[i] - self.hist_iu[i] != 0:
                res[i] = self.hist_iu[i] / float(self.hist_label[i] + self.hist_pre[i] - self.hist_iu[i])
            else:
                res[i] = 0
        print res
        return np.mean(res)


    def _crop(self, img, label):
        assert img.shape[0] == label.shape[0]
        assert img.shape[1] == label.shape[1]
        #print img.shape
        h = np.random.randint(img.shape[0] - 224 + 1, size=1)[0]
        w = np.random.randint(img.shape[1] - 224 + 1, size=1)[0]
        img = img[h:h+224, w:w+224,:]
        label = label[h:h+224, w:w+224]
        return img, label

    def _load_datum(self, id):
        # check cache
        datum = self.db.get(str(id))
        if datum is not None:
            img, label = pickle.loads(datum)
            return img, label
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
        
        return img, label

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
        #datum = datum.transpose((2, 0, 1))
        return datum

    def random_ins(self):
        inx = np.random.randint(len(self.val_list), size=1)[0]
        img, label = self._load_datum(self.val_list[inx])
        return self._crop(img, label), self.val_list[inx]

    def save_label(self, label, name):
        label_rgb_file = osp.join('gen', name + '.png')
        i = self._label_32sc1_to_rgb(label)
        #print i.shape
        imsave(label_rgb_file, i)

    def gen_memory_img(self, mem):
        mi = np.amin(mem)
        ma = np.amax(mem)
        a = (mem - mi) / (ma - mi)
        a = (a * 255).astype(np.uint8)
        name = osp.join('mems', str(self.mem_cnt) + '.png') 
        self.mem_cnt += 1
        imsave(name, a)

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
            img, label = self._crop(img, label)
            batch_data.append(img)
            batch_label.append(label)
        return np.asarray(batch_data, dtype=np.float32), np.asarray(batch_label, dtype=np.float32)

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
            img, label = self._crop(img, label)
            batch_data.append(img)
            batch_label.append(label)
        return np.asarray(batch_data, dtype=np.float32), np.asarray(batch_label, dtype=np.float32)

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
            img, label = self._crop(img, label)
            batch_data.append(img)
            batch_label.append(label)
        return np.asarray(batch_data, dtype=np.float32), np.asarray(batch_label, dtype=np.float32)
