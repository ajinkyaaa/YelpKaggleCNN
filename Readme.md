# Yelp kaggle dataset
Deep Learning


Basic useful feature list:

 * Ctrl+S / Cmd+S to save the file
 * Ctrl+Shift+S / Cmd+Shift+S to choose to save as Markdown or HTML
 * Drag and drop a file into here to load it
 * File contents are saved in the URL so you can share files

Description:- 
At Yelp, there are lots of photos and lots of users uploading photos. These photos provide rich local business information across categories. Teaching a computer to understand the context of these photos is not an easy task. Yelp engineers work on deep learning image classification projects in-house, and you can read about them here. 

In this competition, you are given photos that belong to a business and asked to predict the business attributes. There are 9 different attributes in this problem:

0: good_for_lunch
1: good_for_dinner
2: takes_reservations
3: outdoor_seating
4: restaurant_is_expensive
5: has_alcohol
6: has_table_service
7: ambience_is_classy
8: good_for_kids

These labels are annotated by the Yelp community. Your task is to predict these labels purely from the business photos uploaded by users. 

Since Yelp is a community driven website, there are duplicated images in the dataset. They are mainly due to:

    users accidentally upload the same photo to the same business more than once (e.g., this and this)
    chain businesses which upload the same photo to different branches

Yelp is including these as part of the competition, since these are challenges Yelp researchers face every day. 

To deter hand labeling, Kaggle has supplemented the test set with additional "ignored" businesses. These are not counted in the scoring. 
File descriptions

    train_photos.tgz - photos of the training set
    test_photos.tgz - photos of the test set
    train_photo_to_biz_ids.csv - maps the photo id to business id
    test_photo_to_biz_ids.csv - maps the photo id to business id
    train.csv - main training dataset. Includes the business id's, and their corresponding labels. 
    sample_submission.csv - sample submission and the test dataset. This is the correct format for your predictions. It should include the business_id and the corresponding predicted labels.

 
 
This assignment is divided in three parts:-


** 1) Data Preparation which is done in 	YelpTrainModel-Part1.ipynb. Image data is basically changed to vector format and saved                 in yelpdataset12345.h file.**
 
** 2) YelpTrainModel-Part2.ipynb file is used for training the data using CNN.**
 ================Model Training ===============================
 Train on 164389 samples, validate on 35226 samples
Epoch 1/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5973 - acc: 0.6725 Epoch 00000: val_acc improved from -inf to 0.69199, saving model to weights.00-0.69199.hdf5
164389/164389 [==============================] - 18209s - loss: 0.5973 - acc: 0.6725 - val_loss: 0.5722 - val_acc: 0.6920
Epoch 2/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5713 - acc: 0.6929 Epoch 00001: val_acc improved from 0.69199 to 0.70082, saving model to weights.01-0.70082.hdf5
164389/164389 [==============================] - 17867s - loss: 0.5713 - acc: 0.6929 - val_loss: 0.5606 - val_acc: 0.7008
Epoch 3/20
164300/164389 [============================>.] - ETA: 9s - loss: 0.5611 - acc: 0.7005 Epoch 00002: val_acc improved from 0.70082 to 0.70622, saving model to weights.02-0.70622.hdf5
164389/164389 [==============================] - 18527s - loss: 0.5611 - acc: 0.7005 - val_loss: 0.5522 - val_acc: 0.7062
Epoch 4/20
164300/164389 [============================>.] - ETA: 9s - loss: 0.5543 - acc: 0.7054 Epoch 00003: val_acc did not improve
164389/164389 [==============================] - 18658s - loss: 0.5543 - acc: 0.7054 - val_loss: 0.5551 - val_acc: 0.7041
Epoch 5/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5487 - acc: 0.7093 Epoch 00004: val_acc improved from 0.70622 to 0.71288, saving model to weights.04-0.71288.hdf5
164389/164389 [==============================] - 18363s - loss: 0.5487 - acc: 0.7093 - val_loss: 0.5437 - val_acc: 0.7129
Epoch 6/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5438 - acc: 0.7129 Epoch 00005: val_acc did not improve
164389/164389 [==============================] - 18200s - loss: 0.5438 - acc: 0.7129 - val_loss: 0.5468 - val_acc: 0.7111
Epoch 7/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5391 - acc: 0.7167 Epoch 00006: val_acc improved from 0.71288 to 0.71710, saving model to weights.06-0.71710.hdf5
164389/164389 [==============================] - 18273s - loss: 0.5391 - acc: 0.7167 - val_loss: 0.5379 - val_acc: 0.7171
Epoch 8/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5338 - acc: 0.7205 Epoch 00007: val_acc improved from 0.71710 to 0.71883, saving model to weights.07-0.71883.hdf5
164389/164389 [==============================] - 18120s - loss: 0.5338 - acc: 0.7205 - val_loss: 0.5352 - val_acc: 0.7188
Epoch 9/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5292 - acc: 0.7243 Epoch 00008: val_acc improved from 0.71883 to 0.71999, saving model to weights.08-0.71999.hdf5
164389/164389 [==============================] - 18187s - loss: 0.5291 - acc: 0.7244 - val_loss: 0.5331 - val_acc: 0.7200
Epoch 10/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5234 - acc: 0.7282 Epoch 00009: val_acc improved from 0.71999 to 0.72228, saving model to weights.09-0.72228.hdf5
164389/164389 [==============================] - 18247s - loss: 0.5235 - acc: 0.7281 - val_loss: 0.5308 - val_acc: 0.7223
Epoch 11/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5169 - acc: 0.7328 Epoch 00010: val_acc did not improve
164389/164389 [==============================] - 18384s - loss: 0.5169 - acc: 0.7328 - val_loss: 0.5309 - val_acc: 0.7216
Epoch 12/20
164300/164389 [============================>.] - ETA: 9s - loss: 0.5098 - acc: 0.7382 Epoch 00011: val_acc improved from 0.72228 to 0.72365, saving model to weights.11-0.72365.hdf5
164389/164389 [==============================] - 18427s - loss: 0.5099 - acc: 0.7382 - val_loss: 0.5283 - val_acc: 0.7236
Epoch 13/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.5010 - acc: 0.7445 Epoch 00012: val_acc did not improve
164389/164389 [==============================] - 18401s - loss: 0.5010 - acc: 0.7445 - val_loss: 0.5304 - val_acc: 0.7224
Epoch 14/20
164300/164389 [============================>.] - ETA: 9s - loss: 0.4912 - acc: 0.7516 Epoch 00013: val_acc did not improve
164389/164389 [==============================] - 18522s - loss: 0.4912 - acc: 0.7516 - val_loss: 0.5303 - val_acc: 0.7221
Epoch 15/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.4798 - acc: 0.7595 Epoch 00014: val_acc did not improve
164389/164389 [==============================] - 18158s - loss: 0.4798 - acc: 0.7595 - val_loss: 0.5315 - val_acc: 0.7210
Epoch 16/20
164300/164389 [============================>.] - ETA: 8s - loss: 0.4666 - acc: 0.7687 Epoch 00015: val_acc did not improve
164389/164389 [==============================] - 21974s - loss: 0.4666 - acc: 0.7687 - val_loss: 0.5321 - val_acc: 0.7225
Epoch 17/20
164300/164389 [============================>.] - ETA: 9s - loss: 0.4524 - acc: 0.7776 Epoch 00016: val_acc did not improve
164389/164389 [==============================] - 19021s - loss: 0.4524 - acc: 0.7776 - val_loss: 0.5375 - val_acc: 0.7224
Epoch 18/20
164300/164389 [============================>.] - ETA: 14s - loss: 0.4367 - acc: 0.7877Epoch 00017: val_acc did not improve
164389/164389 [==============================] - 27914s - loss: 0.4367 - acc: 0.7877 - val_loss: 0.5447 - val_acc: 0.7172
Epoch 19/20
164300/164389 [============================>.] - ETA: 9s - loss: 0.4209 - acc: 0.7971 Epoch 00018: val_acc did not improve
164389/164389 [==============================] - 18638s - loss: 0.4209 - acc: 0.7971 - val_loss: 0.5474 - val_acc: 0.7185
Epoch 20/20
164300/164389 [============================>.] - ETA: 9s - loss: 0.4060 - acc: 0.8062 Epoch 00019: val_acc did not improve
164389/164389 [==============================] - 18695s - loss: 0.4061 - acc: 0.8062 - val_loss: 0.5521 - val_acc: 0.7195



==============Model Summary =================================

Model Summary
In [60]:

model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_24 (Conv2D)           (None, 32, 100, 100)      896       
_________________________________________________________________
activation_33 (Activation)   (None, 32, 100, 100)      0         
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 64, 98, 98)        18496     
_________________________________________________________________
activation_34 (Activation)   (None, 64, 98, 98)        0         
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 64, 49, 49)        0         
_________________________________________________________________
dropout_16 (Dropout)         (None, 64, 49, 49)        0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 64, 49, 49)        36928     
_________________________________________________________________
activation_35 (Activation)   (None, 64, 49, 49)        0         
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 64, 47, 47)        36928     
_________________________________________________________________
activation_36 (Activation)   (None, 64, 47, 47)        0         
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 64, 23, 23)        0         
_________________________________________________________________
dropout_17 (Dropout)         (None, 64, 23, 23)        0         
_________________________________________________________________
flatten_6 (Flatten)          (None, 33856)             0         
_________________________________________________________________
dense_10 (Dense)             (None, 512)               17334784  
_________________________________________________________________
activation_37 (Activation)   (None, 512)               0         
_________________________________________________________________
dropout_18 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_11 (Dense)             (None, 5)                 2565      
_________________________________________________________________
activation_38 (Activation)   (None, 5)                 0         
=================================================================
Total params: 17,430,597
Trainable params: 17,430,597
Non-trainable params: 0
================Model Summary =========================


 3) Testing with new data using TestYelp.ipynb.


pred :-

array([[ 0.2015743 ,  0.47074479,  0.57633346,  0.51521689,  0.31590924,
         0.7023387 ,  0.71906191,  0.30924773,  0.53667158]], dtype=float32)
         
finalOutput:-

['Has alchohol', 'Has Table Service']

         



