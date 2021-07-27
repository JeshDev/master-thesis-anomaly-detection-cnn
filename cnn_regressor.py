#!/usr/bin/env python
"""
"""

import tensorflow as tf

print(tf.__version__)

import argparse
from datetime import datetime
from shutil import copy
import time
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from pathlib import Path
from PIL import Image
from PIL import ImageEnhance
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import mobilenet_v2
from cnn_networks import CNNNetworks

# import tensorflow_addons as tfa

__author__ = "Jeshwanth Pilla"
__email__ = "jeshwanth.pilla@siemens.com"
__owner__ = "Clemens Otte"


def get_generators(train_set, test_set, image_dir, batch_size, model_preprocessing_func):
    """

    """
    # Set Validation split(20%) and appropriate pre-processing functions and augmentation techniques
    train_datagen = ImageDataGenerator(validation_split=0.2, horizontal_flip=True, vertical_flip=True,
                                       preprocessing_function=model_preprocessing_func)
    test_datagen = ImageDataGenerator(preprocessing_function=model_preprocessing_func)

    train_generator = train_datagen.flow_from_dataframe(
        train_set,
        directory=image_dir,
        target_size=(128, 128),
        x_col='Image_path',
        y_col='Defect',
        batch_size=batch_size,
        shuffle=False,
        class_mode='raw',
        subset='training')
    validation_generator = train_datagen.flow_from_dataframe(
        train_set,
        directory=image_dir,
        target_size=(128, 128),
        x_col='Image_path',
        y_col='Defect',
        batch_size=batch_size,
        shuffle=False,
        class_mode='raw',
        subset='validation')
    test_generator = test_datagen.flow_from_dataframe(
        test_set,
        directory=image_dir,
        target_size=(128, 128),
        x_col='Image_path',
        y_col='Defect',
        batch_size=batch_size,
        shuffle=False,
        class_mode='raw')

    return train_generator, validation_generator, test_generator


def custom_resnet_preprocess(im):
    """Center each image according to the dataset mean

    """
    if dataset_mean is not None:
        im -= dataset_mean
    return im


def mean_statistics(dataset, folder_path):
    """Center each image according to the dataset mean

    """
    global dataset_mean
    # Convert images in training set into numpy arrays
    dataset_arr = []
    for _file in dataset['Image_path']:
        img = Image.open(os.path.join(folder_path + "/" + _file))
        img_np = np.array(img)
        dataset_arr.append(img_np)
    dataset_arr = np.array(dataset_arr)

    # Flatten Image arrays and compute statistics
    dataset_flat = dataset_arr.flatten()
    dataset_mean = np.mean(dataset_flat)
    dataset_min = np.min(dataset_flat)
    dataset_max = np.max(dataset_flat)
    dataset_std = np.std(dataset_flat)

    print('Dataset - Mean: %f, Min: %f, Max: %f, Std: %f' % (dataset_mean, dataset_min, dataset_max, dataset_std))

    # Create histogram of the training set
    # create_training_hist(dataset_flat)


def create_training_hist(train_set_flat):
    """Center each image according to the dataset mean

    """
    plot5 = plt.figure(5)
    n, bins, patches = plt.hist(train_set_flat, density=True, bins=range(256))
    plt.title('Training Set Histogram')
    plt.xlabel('Gray Value Distribution')
    plt.xlim(0, 255)
    cm = plt.cm.get_cmap('cool')
    norm = colors.Normalize(vmin=bins.min(), vmax=bins.max())
    for b, p in zip(bins, patches):
        p.set_facecolor(cm(norm(b)))
    plt.savefig('Training_set_Histogram.png', bbox_inches='tight')
    plt.clf()


def publish_top_errors(prediction_df, image_path, phase, model_path):  # raw_df):
    """Center each image according to the dataset mean

    """
    dir_name = 'top_train_error_images' if phase == 'Train' else 'top_test_error_images'
    # deviation_files = []

    dir_path = model_path + dir_name
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    for index, row in prediction_df.iterrows():
        if row['Deviation'] > 0.2 or row['Deviation'] < -0.2:
            # deviation_files.append(row['Image'])
            folder_name = str(round(row['Deviation'], 3)) + '_' + str(round(row['Defect Score'], 3)) + '_' + \
                          row['Image'].replace(".png", "").split("/")[1]
            os.mkdir(os.path.join(dir_path, folder_name))
            copy(os.path.join(image_path, row['Image']), os.path.join(dir_path, folder_name))
            parent_image = row['Image'].split("/")[0] + '.jpeg'
            copy(os.path.join(image_path.replace("/patches", ""), parent_image), os.path.join(dir_path, folder_name))

    # filtered_df = raw_df[~raw_df['Image_path'].isin(deviation_files)]
    # filtered_df.to_csv('overview_filtered_auto.csv', sep=',', index = False)
    # return filtered_df


def print_top_deviations(cnn_model, generator, dataset, phase, model_path):
    """Center each image according to the dataset mean

    """
    scores = cnn_model.predict(generator)

    if phase == "train":
        pred_df = pd.DataFrame({'ImageName': generator.filenames, 'Predicted Score': scores[:, 0]})
        temp_df = pd.merge(dataset.rename(columns={'Image_path': 'Image', 'Defect': 'Defect Score'}),
                           pred_df,
                           left_on='Image',
                           right_on='ImageName')
        prediction_df = temp_df[['Image', 'Defect Score', 'Predicted Score']]
    else:
        prediction_df = pd.DataFrame(
            {'Image': generator.filenames, 'Defect Score': dataset['Defect'], 'Predicted Score':
                scores[:, 0]})

    prediction_df['Deviation (Absolute)'] = abs(prediction_df['Defect Score'] - prediction_df['Predicted Score'])
    prediction_df['Deviation'] = prediction_df['Defect Score'] - prediction_df['Predicted Score']
    prediction_df.sort_values(by=['Deviation'], ascending=False, inplace=True)
    prediction_df.to_excel(model_path + phase + '_predictions.xlsx', index=False)

    return prediction_df


def main(args):
    raw_df = pd.read_csv(args.csv_file, sep=",")
    # r = filter_files(raw_df)
    # sys.exit()
    labelled_imageset = raw_df[['Image_path', 'Defect']]
    print(f"Total {(len(labelled_imageset))} image patches are found ")
    train_set_complete, test_set = train_test_split(labelled_imageset, test_size=0.2,
                                                    random_state=42)  # test split = 20%
    print(f"Total {len(train_set_complete)} image patches are present in training and validation set ")

    # Evaluate mean statistics of whole dataset
    mean_statistics(labelled_imageset, args.image_patches_path)

    df = pd.read_excel('mob_test_predictions.xlsx', engine='openpyxl', usecols="A:C")
    df2 = df.rename({'Predicted Score': 'MobileNet Prediction'}, axis='columns')
    df3 = pd.read_excel('res_test_predictions.xlsx', engine='openpyxl', usecols="A:C")
    df4 = df3.rename({'Predicted Score': 'ResNet Prediction'}, axis='columns')
    predictions_df = pd.merge(df2, df4, on=['Image', 'Defect Score'])

    predictions_df.to_excel('prediction_comparison.xlsx', index=False)
    plot1 = plt.figure(1)
    ax1 = plot1.add_subplot(111)
    ax1.scatter(predictions_df['Defect Score'], predictions_df['ResNet Prediction'], alpha=0.3, color='tab:orange',
                label='ResNet')
    ax1.scatter(predictions_df['Defect Score'], predictions_df['MobileNet Prediction'], alpha=0.3, color='tab:blue',
                label='MobileNet')
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Prediction")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    # plt.title('Comparison of ResNet50 and MobileNetV2 Predictions on Test Set')
    plt.savefig('Model_comparison.png', bbox_inches='tight')
    # ax2 = plot1.add_subplot(111)
    # ax2.scatter(predictions_df['ResNet Prediction'], predictions_df['MobileNet Prediction'], alpha=0.3,
    #           color='tab:orange',
    #           label='Prediction')
    # ax2.set_xlabel("ResNet50 Prediction")
    # ax2.set_ylabel("MobileNetV2 Prediction")
    # plt.title('Agreement of ResNet50 and MobileNetV2 Predictions on Test Set')
    # plt.savefig('Models_agreement2.png', bbox_inches='tight')
    plt.show()

    correlation_matrix = np.corrcoef(predictions_df['ResNet Prediction'], predictions_df['MobileNet Prediction'])
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    print(r_squared)
    exit()

    count = 0
    # for factor in [0.8, 0.9, 1.1, 1.2]:
    #    train_set_aug_arr = create_aug_train(factor, train_set_complete, args.image_patches_path)
    #    train_set_aug_df = pd.DataFrame(train_set_aug_arr)
    #   count += len(train_set_aug_df)
    #   train_set_complete = train_set_complete.append(train_set_aug_df)
    # print("Total %s augmented image patches are added to training and validation set " % count)
    # train_set_complete.to_csv(os.path.join(args.image_patches_path + '/' + 'overview_train_aug.csv'), sep=',',
    #                         index=False)

    # create_augmented_test_set(test_set, args.image_patches_path)
    # for factor in [0.8, 0.9, 1.1, 1.2]:
    # create_aug_test(factor, test_set, args.image_patches_path)

    # Exclude validation set from trainset
    train_set = train_set_complete.head(int(len(train_set_complete) * (80 / 100)) + 1)

    # Set input preprocessing function according to the network chosen
    if args.model == 1:
        model_name = 'ResNet50'
        model_preprocessing_func = custom_resnet_preprocess
    else:
        model_name = 'MobileNet'
        model_preprocessing_func = mobilenet_v2.preprocess_input

    # Create Data Generators for the image patches
    train_generator, validation_generator, test_generator = get_generators(train_set_complete, test_set,
                                                                           args.image_patches_path,
                                                                           args.batch_size,
                                                                           model_preprocessing_func)

    if args.saved_model_folder is None:

        # Create Model of chosen network
        if args.model == 1:
            cnn_model = CNNNetworks.create_resnet(args.learning_rate, load_pretrained=True, top='max',
                                                  loss_function='mse')
            cnn_model2 = CNNNetworks.create_mobilenet(args.learning_rate, load_pretrained=True, top='max',
                                                      loss_function='mse')
        else:
            cnn_model = CNNNetworks.create_mobilenet(args.learning_rate, load_pretrained=True, top='max',
                                                     loss_function='mse')

        # cnn_model.summary()

        # Logs for Tensorboard
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S').replace(":", "-")
        log_dir = "logs/" + "BS-" + str(args.batch_size) + " EP-" + str(args.no_of_epochs) + " LR-" + \
                  str(args.learning_rate) + " " + timestamp.replace(" ", "_")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Fit model
        model_history = cnn_model.fit(train_generator, validation_data=validation_generator,
                                      epochs=args.no_of_epochs,
                                      callbacks=[tensorboard_callback])  # verbose= 2, steps_per_epoch=10,
        model_history2 = cnn_model2.fit(train_generator, validation_data=validation_generator,
                                        epochs=args.no_of_epochs,
                                        callbacks=[tensorboard_callback])  # verbose= 2, steps_per_epoch=10,

        # Save model
        model_dir_name = model_name + "_" + "BS-" + str(args.batch_size) + "_EP-" + str(args.no_of_epochs) + "_LR-" + \
                         str(args.learning_rate) + "_" + timestamp.replace(" ", "_")
        model_path = './' + model_dir_name + '/'
        cnn_model.save(model_dir_name)

        # Save the loss curves
        # plot1 = plt.figure(1)
        # plt.plot(model_history.history['loss'], label='training loss')
        # plt.plot(model_history.history['val_loss'], label='validation loss')
        # plt.title('Loss - ' + model_name)
        # plt.ylabel('Epoch loss')
        # plt.xlabel('Step')
        # plt.legend(loc="upper right")
        # plt.savefig(model_path + 'losscurves.png', bbox_inches='tight')
        # plt.close()

    else:

        # Load from saved model (if exists)
        model_path = './' + args.saved_model_folder + '/'
        model_path2 = './' + args.saved_model_folder2 + '/'
        saved_model = load_model(model_path)
        saved_model.compile(optimizer=Adam(lr=args.learning_rate), loss='mse', metrics=['mse', 'mae', 'mape'])
        cnn_model = saved_model
        saved_model2 = load_model(model_path2)
        saved_model2.compile(optimizer=Adam(lr=args.learning_rate), loss='mse', metrics=['mse', 'mae', 'mape'])
        cnn_model2 = saved_model2

    # Print the scores for test set
    # test_prediction_df = print_top_deviations(cnn_model, test_generator, test_set, "test", model_path)

    # Print the scores for train set
    # train_prediction_df = print_top_deviations(cnn_model, train_generator, train_set, "train", model_path)

    # Plot level of agreement between two models
    plot_compare_models(cnn_model, cnn_model2, test_generator, test_set, model_path)
    exit()

    # Filter highly deviating labels (or noise)
    # filter_noise(raw_df, test_prediction_df, train_prediction_df)

    # Publish plots to visualize model performance and accuracy
    # publish_plots(test_prediction_df, train_prediction_df, model_path)

    # Publish images with bad test and training prediction scores
    # publish_top_errors(test_prediction_df, args.image_patches_path, "Test", model_path)  # , raw_df)
    # publish_top_errors(train_prediction_df, args.image_patches_path, "Train", model_path)  # , test_filtered_df)

    # Evaluate Model
    test_errors = cnn_model.evaluate(test_generator)  # steps =
    print("MSE Score on test-set: ", test_errors[1])

    # Evaluate Model for augmented test sets
    print_aug_test_set_metrics(args.image_patches_path, model_preprocessing_func, cnn_model, args.batch_size)

    # Calculate Pearson Correlation Coefficient for the trainset and testset predictions
    # test_corr = tfp.stats.correlation(test_prediction_df['Defect Score'], test_prediction_df['Predicted Score'],
    #                                  sample_axis=0, event_axis=None)
    # train_corr = tfp.stats.correlation(train_prediction_df['Defect Score'], train_prediction_df['Predicted Score'],
    #                                   sample_axis=0, event_axis=None)
    tf.print("Pearson Correlation for test-set: ", test_corr)
    tf.print("Pearson Correlation for train-set: ", train_corr)

    # Calculate R Square for the train-set and test-set predictions
    # rsquare = tfa.metrics.RSquare(dtype=tf.float32)
    # rsquare.update_state(test_prediction_df['Defect Score'], test_prediction_df['Predicted Score'])
    # print('R^2 score for test-set is: ', rsquare.result().numpy())
    # rsquare.update_state(train_prediction_df['Defect Score'], train_prediction_df['Predicted Score'])
    print('R^2 score for train-set is: ', rsquare.result().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='trains ConvNet model on powder-bed images')
    parser.add_argument('csv_file',
                        help='path to csv file containing the scores')
    parser.add_argument('image_patches_path',
                        help='folder path containing powderbed image patches')
    parser.add_argument('-m', dest='model',
                        type=int,
                        default=1,
                        help='The Convolution network you would like to choose '
                             '1: ResNet50'
                             '2: MobileNetV2)')
    parser.add_argument('-e', dest='no_of_epochs',
                        type=int,
                        default=1,
                        help='Number of epochs to train the model. An epoch is an iteration over the entire '
                             'x and y data provided.')
    parser.add_argument('-b', dest='batch_size',
                        type=int,
                        default=16,
                        help='Number of samples per batch of computation')
    parser.add_argument('-l', dest='learning_rate',
                        type=float,
                        default=0.001,
                        help='floating point value of Learning Rate')
    parser.add_argument('-s', dest='saved_model_folder',
                        help='folder name containing the saved model')
    parser.add_argument('-s2', dest='saved_model_folder2',
                        help='folder name containing the saved model 2')
    main(parser.parse_args())
