import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import svm, metrics, pipeline
import time
from sklearn import preprocessing
om sklearn import preprocessing
from sklearn.kernel_approximation import (RBFSampler, Nystroem)



df = pd.read_csv(r"/beegfs/jn1664/fer2013/fer2013.csv")
train_df = df[df['Usage']=='Training']
val_df = df[df['Usage']=='PublicTest']
test_df = df[df['Usage']=='PrivateTest']
print(train_df.shape, val_df.shape, test_df.shape)

Y_train = np.array(train_df['emotion'], np.float)
Y_val = np.array(val_df['emotion'], np.float)
Y_test = np.array(test_df['emotion'], np.float)
print(Y_train.shape, Y_val.shape, Y_test.shape)


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
def get_images(mode):
    if mode=='train':
        imagebuffer = np.array(train_df['pixels'])
    elif mode=='val':
        imagebuffer = np.array(val_df['pixels'])
    elif mode=='test':
        imagebuffer = np.array(test_df['pixels'])
    images = np.array([np.fromstring(image,np.uint8,sep=' ') for image in imagebuffer])
    del imagebuffer
    print(images.shape)
    return images




# We learn the images
images_train = get_images('train')
scaler = preprocessing.StandardScaler().fit(images_train)
images_train = scaler.transform(images_train)
#print(scaler.mean_, scaler.scale_)




# create pipeline from kernel approximation
# and linear svm
feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", svm.LinearSVC())])

nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("svm", svm.LinearSVC())])




# fit and predict using linear svm:
linear_svm = svm.LinearSVC(C=0.1)
start_t = time.time()
linear_svm.fit(images_train, Y_train)
end_t = time.time()
print('[Total Train Time]: {:.4f}'.format(end_t-start_t))




# Now predict the emotion of the images
images_val = get_images('val')
images_val = scaler.transform(images_val)

start_t = time.time()
predicted_val = linear_svm.predict(images_val)
end_t = time.time()
linear_svm_score = linear_svm.score(images_val, Y_val)
print('[Total Validation Time]: {:.4f}'.format(end_t-start_t))

print("Classification report for classifier %s:\n%s\n"
      % (linear_svm, metrics.classification_report(Y_val, predicted_val)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_val, predicted_val))
print('Validation Score:{}'.format(linear_svm_score))






images_test = get_images('test')
images_test = scaler.transform(images_test)

start_t = time.time()
predicted_test = linear_svm.predict(images_test)
end_t = time.time()
linear_svm_score = linear_svm.score(images_test, Y_test)
print('[Total Test Time]: {:.4f}'.format(end_t-start_t))

print("Classification report for classifier %s:\n%s\n"
      % (linear_svm, metrics.classification_report(Y_test, predicted_test)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, predicted_test))
print('Test Score:{}'.format(linear_svm_score))



sample_sizes = 30 * np.arange(1, 10)
fourier_scores = []
fourier_times = []

for D in sample_sizes:
    fourier_approx_svm.set_params(feature_map__n_components=D)
    start = time.time()
    fourier_approx_svm.fit(images_train, Y_train)
    fourier_times.append(time.time() - start)

    fourier_score = fourier_approx_svm.score(images_test, Y_test)
    fourier_scores.append(fourier_score)

print(fourier_scores)
