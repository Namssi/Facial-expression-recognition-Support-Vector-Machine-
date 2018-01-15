import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import time
from sklearn import preprocessing



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




# Create a classifier: a support vector classifier
classifier = svm.SVC(C=100, gamma=1)




# We learn the images
images_train = get_images('train')
scaler = preprocessing.StandardScaler().fit(images_train)
images_train = scaler.transform(images_train)

start_t = time.time()
classifier.fit(images_train, Y_train)
end_t = time.time()
print('[Total Train Time]: {:.4f}'.format(end_t-start_t))




# Now predict the emotion of the images
images_val = get_images('val')
images_val = scaler.transform(images_val)

start_t = time.time()
predicted_val = classifier.predict(images_val)
end_t = time.time()
print('[Total Validation Time]: {:.4f}'.format(end_t-start_t))

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(Y_val, predicted_val)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_val, predicted_val))
print('Validation Score:{}'.format(classifier.score(images_val, Y_val)))







images_test = get_images('test')
images_test = scaler.transform(images_test)

start_t = time.time()
predicted_val = classifier.predict(images_test)
end_t = time.time()
print('[Total Test Time]: {:.4f}'.format(end_t-start_t))

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(Y_test, predicted_val)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, predicted_val))
print('Test Score:{}'.format(classifier.score(images_test, Y_test)))


