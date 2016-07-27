from fcn.fcn8_vgg import FCN8VGG
from fcn.loss import loss
from pascal import SegmentationClassDataset
import numpy as np
import tensorflow as tf
import sys, time
from memory import Memory

p={'epochs':201,'momentum':.9,'minibatchSz':30,'reportMinibatch':True,'resumeFname':'','v':3,'outStream':sys.stderr,'lr':0.000001,'lrMult':None,'regL2':0.0,'regL1':0.0,'minibatchesPerEpoch':50, 'valFrequence':10, 'lr_decay':0.96}
batch_size = p['minibatchSz']
img_height = 224
img_width = 224
img_channel = 3
num_classes = 21
images = tf.placeholder(tf.float32,shape=(batch_size,img_height, img_width, img_channel))
labels = tf.placeholder(tf.int64,shape=(batch_size*img_height*img_width))
learning_rate = tf.placeholder(tf.float32, shape=[])
#valid_index = tf.placeholder(tf.bool,shape=(batch_size*img_height*img_width))

vgg_fcn = FCN8VGG()
with tf.name_scope("content_vgg"):
    upscore32 = vgg_fcn.build(images, num_classes=num_classes, debug=False, random_init_fc8=True, train=True)
    upscore32_test = vgg_fcn.build(images, num_classes=num_classes, debug=False, random_init_fc8=True, train=False)

mm = Memory()
with tf.name_scope("memory"):
   mems = mm.build(images, batch_size=batch_size)
   pred_score = mm.use_memory(upscore32, mems, num_classes=num_classes, kernal_size=3, train=True, wd=1e-3)
   pred_score_test = mm.use_memory(upscore32_test, mems, num_classes=num_classes, kernal_size=3, train=False, wd=1e-3)

cc = loss(pred_score, labels,num_classes=num_classes)
cc_test = loss(pred_score_test, labels,num_classes=num_classes)
pred_test = tf.argmax(pred_score_test, dimension=3)

optimizer=tf.train.AdamOptimizer(learning_rate)
optimizer2 = tf.train.AdamOptimizer(1e-3, beta1=0.5)
varis=tf.trainable_variables()
varis_cnn = []
varis_lstm = []
for i,v in enumerate(varis):
    print v.name
    if v.name.startswith("sensor") or v.name.startswith("lstm") or v.name.startswith("writeW"):
        varis_lstm.append(v)
    else:
        varis_cnn.append(v)
grads_cnn=optimizer.compute_gradients(cc, varis_cnn)
grads_lstm=optimizer2.compute_gradients(cc, varis_lstm)
for i,(g,v) in enumerate(grads_cnn):
    if g is not None:
        grads_cnn[i]=(tf.clip_by_norm(g,10),v)
for i,(g,v) in enumerate(grads_lstm):
    if g is not None:
        grads_lstm[i]=(tf.clip_by_norm(g,10),v)
train_op1=optimizer.apply_gradients(grads_cnn)
train_op2=optimizer2.apply_gradients(grads_lstm)
train_op = tf.group(train_op1, train_op2)

fetches=[cc,train_op]
fetches_val=[cc_test, pred_test, mems]
fetches_gen=[pred_test]
print('Finished building Network')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.InteractiveSession(config=config)
tf.initialize_all_variables().run()
saver = tf.train.Saver()
#saver.restore(sess,"my-model-190")
dataset = SegmentationClassDataset()
print('Finish initializing')

print "start training"
e=p['outStream']
for epoch in range(p['epochs']):

    train_err = 0
    train_batches = 0
    start_time = time.time()
    if epoch >= 60:
        lr = p['lr']*(p['lr_decay']**((epoch-60)*1.0/10.0))
    else:
        lr = p['lr']
    if p['v']>1:
        e.write('Epoch '+str(epoch)+': \n')
    for minibatchCount in range(p['minibatchesPerEpoch']):
        if p['v']>2:
            e.write('.')
        inputs, targets = dataset.next_train_batch(p['minibatchSz'])
        targets = np.reshape(targets, (-1))
        feed_dict={images:inputs, labels:targets, learning_rate:lr}
        loss,_ = sess.run(fetches,feed_dict)
        train_err += loss
        if p['reportMinibatch']:
            #verr, vacc = val_fn(inputs, targets, valid_targets)
            #print 'E:%3d  B:%4d  Acc:%.3f  Err:%8.4f'%(epoch,bc,vacc,verr)
            print 'E:%3d  B:%4d  Loss:%8.8f'%(epoch, minibatchCount, loss)
            #print mems[0][0][:20][:]
        train_batches += 1
    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    miou = 0
    if epoch % p['valFrequence'] == 0:
        if p['v']>1:
            e.write('\nValidating: ')
        hist_gt = [0] * 21
        hist_pr = [0] * 21
        hist_iou = [0] * 21
        res = [0] * 21
        equal_sum = 0
        all_sum = 0
        for minibatchCount in range(p['minibatchesPerEpoch']):
            if p['v']>2:
                e.write('.')
            inputs, targets = dataset.next_val_batch(p['minibatchSz'])
            targets = np.reshape(targets, (-1))
            valid_targets = (targets != -1)
            feed_dict={images:inputs, labels:targets}
            loss,pred,mems = sess.run(fetches_val,feed_dict)
            if minibatchCount == 0:
                for icnt in range(10):
                    dataset.gen_memory_img(mems[icnt].reshape([img_height, img_width]))
            pred = pred.reshape([-1])
            equal_sum += np.sum(np.equal(pred[valid_targets], targets[valid_targets]))
            all_sum += np.sum(valid_targets)
            #acc = np.mean(np.equal(pred[valid_targets], targets[valid_targets]))
            for cls in range(21):
                mask_1 = (targets[valid_targets] == cls)
                mask_2 = (pred[valid_targets] == cls)
                hist_gt[cls] += np.sum(mask_1)
                hist_pr[cls] += np.sum(mask_2)
                hist_iou[cls] += np.sum(mask_2 & mask_1)
            val_err += loss
            #val_acc += acc
            val_batches += 1
        for cls in range(21):
            res[cls] = float(hist_iou[cls]) / (hist_pr[cls] + hist_gt[cls] - hist_iou[cls])
        miou = np.mean(res)
        val_acc = float(equal_sum) / all_sum
        if p['v']>2:
            e.write('\n')
            print res
        # dataset.reset_test()
        # for i in range(150):
        #     shape = dataset.test_size()
        #     for ii in range(shape[0]):
        #         for jj in range(shape[1]):
        #             img_crop = dataset.test_crop(ii, jj)
        #             [pre_label] = sess.run(fetches_gen, {images:img_crop})
        #             dataset.cal_crop_hist(pre_label[0], ii, jj)
        # miou = dataset.cal_iou()
        # dataset.shuffle_test()
        saver.save(sess, 'mem-model', global_step=epoch)

    if p['v']>0:
        e.write("Epoch {} of {} took {:.3f}s\n".format(epoch + 1, p['epochs'], time.time() - start_time))
        e.write("  training loss:\t\t{:.6f}     lr:\t\t{:.8f}\n".format(train_err / train_batches, lr))
        if epoch % p['valFrequence'] == 0:
            e.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
            e.write("  validation accuracy:\t\t{:.2f} %\n".format(val_acc  * 100))
            e.write("  validation miou:\t\t{:.6f} %\n".format(miou * 100))
    if p['v']>3 and len(p['resumeFname'])>0:
        cPickle.dump({'model':model,'epoch':epoch,'params':p},open(p['resumeFname'],'w'))
