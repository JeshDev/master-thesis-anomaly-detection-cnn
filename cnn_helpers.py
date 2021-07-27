import os


def create_augmented_test_set(test_set, folder_path):
    test_set_aug_arr = []
    dir_path = folder_path + '/augmented_test_set'
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    for index, row in test_set.iterrows():
        filepath = row['Image_path']
        filename = row['Image_path'].split('/')[1].replace('.png', '')
        filename_png = row['Image_path'].split('/')[1]
        test_set_aug_arr.append(
            {'Image': filename_png, 'Defect': row['Defect'], 'Type': 'Original', 'Image_path': row['Image_path']})
        img = Image.open(os.path.join(folder_path + "/" + filepath))
        os.mkdir(os.path.join(dir_path, filename))
        copy(os.path.join(folder_path, row['Image_path']), os.path.join(dir_path, filename))
        for factor in [0.8, 0.9, 1.1, 1.2]:
            contrast_enhancer = ImageEnhance.Contrast(img)
            bright_enhancer = ImageEnhance.Brightness(img)
            img_contrast_en = contrast_enhancer.enhance(factor)
            img_contrast_en.save(os.path.join(dir_path + '/' + filename, filename + '_ct_' + str(factor) + '.png'))
            img_bright_en = bright_enhancer.enhance(factor)
            img_bright_en.save(os.path.join(dir_path + '/' + filename, filename + '_br_' + str(factor) + '.png'))
            img_contrast_en_path = 'augmented_test_set' + '/' + filename + '/' + filename + '_ct_' + str(
                factor) + '.png'
            img_bright_en_path = 'augmented_test_set' + '/' + filename + '/' + filename + '_br_' + str(factor) + '.png'
            test_set_aug_arr.append(
                {'Image': filename_png, 'Defect': row['Defect'], 'Type': 'Contrast_' + str(factor),
                 'Image_path': img_contrast_en_path})
            test_set_aug_arr.append(
                {'Image': filename_png, 'Defect': row['Defect'], 'Type': 'Brightness_' + str(factor),
                 'Image_path': img_bright_en_path})
    test_set_aug_df = pd.DataFrame(test_set_aug_arr)
    test_set_aug_df.to_csv(os.path.join(folder_path.replace('patches', '') + 'overview_test_aug.csv'), sep=',', index=False)
    print("Total %s augmented patches are created from %s patches " % (len(test_set_aug_df.index), len(test_set)))
    return test_set_aug_df


def generate_scores_aug_test_set(aug_csv, image_dir, model_preprocessing_func, cnn_model, batch_size, model_path):
    aug_test_df = pd.read_csv(aug_csv, sep=",")
    aug_test_datagen = ImageDataGenerator(preprocessing_function=model_preprocessing_func)
    aug_test_generator = aug_test_datagen.flow_from_dataframe(
        aug_test_df,
        directory=image_dir,
        target_size=(128, 128),
        x_col='Image_path',
        y_col='Defect',
        batch_size=batch_size,
        shuffle=False,
        class_mode='raw')
    scores = cnn_model.predict(aug_test_generator)
    aug_prediction_df = pd.DataFrame(
        {'Aug_Image': aug_test_generator.filenames, 'Image': aug_test_df['Image'], 'Type': aug_test_df['Type'],
         'Defect Score': aug_test_df['Defect'], 'Predicted Score':
             scores[:, 0]})
    aug_prediction_df.to_excel(model_path + 'aug_test_predictions.xlsx', index=False)
    aug_grpby = aug_prediction_df.groupby(['Image', 'Type'])
    print(aug_grpby)
    aug_grpby.to_excel(model_path + 'aug_test_predictions_grpby.xlsx', index=False)
    #rmse = tf.keras.metrics.RootMeanSquaredError()
    #rmse.update_state(train_prediction_df['Defect Score'], train_prediction_df['Predicted Score'])
    #print('R^2 score for train-set is: ', rsquare.result().numpy())


def filter_files(raw_df):
    r = []
    rootdir = './noise_labels'
    for dirs in os.listdir(rootdir):
        for dir in os.listdir(rootdir + '/' + dirs):
            image = dir.rsplit('_', 2)[0].split('_', 2)[2]
            patch = dir.split('_', 2)[2] + '.png'
            r.append(image + "/" + patch)
    filtered_df = raw_df[~raw_df['Image_path'].isin(r)]
    filtered_df.to_csv('overview_ex6_filtered_Iter_3.csv', sep=',', index=False)
    return r


def filter_noise(raw_df, test_predict_df, train_predict_df):
    threshold = -0.2000
    high_dev_test = test_predict_df.loc[test_predict_df['Deviation'] <= threshold]
    print('No. of deviations of test-set predictions more than %s: %d' % (threshold, len(high_dev_test.index)))
    high_dev_train = train_predict_df.loc[train_predict_df['Deviation'] < threshold]
    print('No. of deviations of train-set predictions more than %s: %d' % (threshold, len(high_dev_train.index)))
    temp_filter_df = raw_df[~raw_df['Image_path'].isin(high_dev_test['Image'])]
    filtered_df = temp_filter_df[~temp_filter_df['Image_path'].isin(high_dev_train['Image'])]
    filtered_df.to_csv('overview_ex6_filtered.csv', sep=',', index=False)
    return filtered_df

def plot_compare_models(resnet_model, mobilenet_model, generator, dataset, model_path):
    resnet_scores = resnet_model.predict(generator)
    mobilenet_scores = mobilenet_model.predict(generator)

    # prediction_df = pd.DataFrame(
    #     {'Image': generator.filenames, 'Defect Score': dataset['Defect'], 'ResNet Prediction':
    #         resnet_scores[:, 0]})
    # prediction_df2 = pd.DataFrame(
    #     {'Image': generator.filenames, 'Defect Score': dataset['Defect'], 'MobileNet Prediction':
    #         mobilenet_scores[:, 0]})
    # predictions_df = pd.merge(prediction_df, prediction_df2, on=['Image', 'Defect Score'])
    # predictions_df.to_excel('prediction_comparison.xlsx', index=False)
    # print(predictions_df)
    # predictions_df = pd.DataFrame(
    #      {'Image': generator.filenames, 'Defect Score': dataset['Defect'], 'ResNet Prediction': resnet_scores[:, 0],
    #       'MobileNet Prediction': mobilenet_scores[:, 0]
    #       })
    # print(predictions_df)

    # Plot the scatter plot for Actual vs prediction of test set
    plot6 = plt.figure(6)  # , figsize=(10, 4.8)
    # ax1 = predictions_df.plot(kind='scatter', x='Defect Score', y='ResNet Prediction', alpha=0.3, color='tab:orange',
    #                            label='ResNet')
    # plot7 = plt.figure(7)
    # ax2 = predictions_df.plot(kind='scatter', x='Defect Score', y='MobileNet Prediction', alpha=0.3, color='tab:blue',
    #                      label='MobileNet', ax=ax1)
    ax1 = plot6.add_subplot(111)
    ax1.scatter(predictions_df['Defect Score'], predictions_df['ResNet Prediction'], alpha=0.3, color='tab:orange',
                label='ResNet')
    ax1.scatter(predictions_df['Defect Score'], predictions_df['MobileNet Prediction'], alpha=0.3, color='tab:blue',
                label='MobileNet')
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Prediction")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.title('Comparison of ResNet50 and MobileNetV2 Predictions on Test Set')
    plt.savefig(model_path + 'Model_comparison.png', bbox_inches='tight')
    plt.show()

    plot7 = plt.figure(7)  # , figsize=(10, 4.8)
    ax2 = plot7.add_subplot(111)
    ax2.scatter(predictions_df['ResNet Prediction'], predictions_df['MobileNet Prediction'], alpha=0.3,
                color='tab:orange',
                label='Prediction')
    ax2.scatter(predictions_df['ResNet Prediction'], predictions_df['Defect Score'], alpha=0.3, color='tab:blue',
                label='Ground Truth')
    ax2.set_xlabel("ResNet Prediction")
    ax2.set_ylabel("MobileNet Prediction")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.title('Agreement of ResNet50 and MobileNetV2 Predictions on Test Set')
    plt.savefig(model_path + 'Models_agreement.png', bbox_inches='tight')
    plt.show()

    return predictions_df

def publish_plots(test_df, train_df, model_path):
    # Plot the scatter plot for Actual vs prediction of test set
    x = test_df['Defect Score']
    y = test_df['Predicted Score']
    plot2 = plt.figure(2)
    plt.scatter(x, y, alpha=0.3)
    plt.title('Ground Truth vs Prediction - Test Set')
    plt.ylabel('Predicted Score')
    plt.xlabel('Ground Truth')
    plt.savefig(model_path + 'Test Scores Scatterplot.png', bbox_inches='tight')
    # plt.show()

    # Plot a histogram for Error deviation in test set
    plot3 = plt.figure(3)
    plt.hist(test_df['Deviation'], density=True, bins=300)
    plt.title('Deviation - Test Set')
    plt.xlabel('Test Error Deviation')
    plt.savefig(model_path + 'Test Error Histogram.png', bbox_inches='tight')
    # plt.show()

    # Plot the scatter plot for Actual vs prediction of training set
    plot4 = plt.figure(4)
    x2 = train_df['Defect Score']
    y2 = train_df['Predicted Score']
    plt.scatter(x2, y2, alpha=0.3)
    plt.title('Ground Truth vs Prediction - Training Set')
    plt.ylabel('Predicted Score')
    plt.xlabel('Ground Truth')
    plt.savefig(model_path + 'Train Scores Scatterplot.png', bbox_inches='tight')
    # plt.show()

    # Plot a histogram for Error deviation in train set
    plot5 = plt.figure(5)
    plt.hist(train_df['Deviation'], density=True, bins=300)
    plt.title('Deviation - Training Set')
    plt.xlabel('Train Error Deviation')
    plt.savefig(model_path + 'Train Error Histogram.png', bbox_inches='tight')
    # plt.show()


def create_aug_test(factor, test_set, folder_path):
    dir_path = folder_path + '/augmented_test_set_new'
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    aug_cr_folder_path = dir_path + '/contrast_' + str(factor)
    aug_br_folder_path = dir_path + '/brightness_' + str(factor)
    Path(aug_cr_folder_path).mkdir(parents=True, exist_ok=True)
    Path(aug_br_folder_path).mkdir(parents=True, exist_ok=True)
    test_set_aug_br_arr = []
    test_set_aug_cr_arr = []
    for index, row in test_set.iterrows():
        filepath = row['Image_path']
        filename = row['Image_path'].split('/')[1]
        img = Image.open(os.path.join(folder_path + "/" + filepath))
        contrast_enhancer = ImageEnhance.Contrast(img)
        bright_enhancer = ImageEnhance.Brightness(img)
        img_contrast_en = contrast_enhancer.enhance(factor)
        img_contrast_en.save(aug_cr_folder_path + '/' + filename)
        img_bright_en = bright_enhancer.enhance(factor)
        img_bright_en.save(aug_br_folder_path + '/' + filename)
        aug_cr_image_path = '/augmented_test_set_new' + '/contrast_' + str(factor) + '/'
        aug_br_image_path = '/augmented_test_set_new' + '/brightness_' + str(factor) + '/'
        test_set_aug_cr_arr.append(
            {'Image': filename, 'Defect': row['Defect'], 'Image_path': aug_cr_image_path + filename})
        test_set_aug_br_arr.append(
            {'Image': filename, 'Defect': row['Defect'], 'Image_path': aug_br_image_path + filename})
    test_set_aug_cr_df = pd.DataFrame(test_set_aug_cr_arr)
    test_set_aug_br_df = pd.DataFrame(test_set_aug_br_arr)
    test_set_aug_cr_df.to_csv(os.path.join(aug_cr_folder_path + '/' + 'overview_test_aug_cr_' + str(factor) + '.csv'),
                              sep=',', index=False)
    test_set_aug_br_df.to_csv(os.path.join(aug_br_folder_path + '/' + 'overview_test_aug_br_' + str(factor) + '.csv'),
                              sep=',', index=False)


def create_aug_train(factor, train_set, folder_path):
    dir_path = folder_path + '/augmented_train_set'
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    sampled_train_set = train_set.sample(frac=0.25 / (4 * 2))
    train_set_aug_arr = []
    for index, row in sampled_train_set.iterrows():
        filepath = row['Image_path']
        filename = row['Image_path'].split('/')[1]
        img = Image.open(os.path.join(folder_path + "/" + filepath))
        contrast_enhancer = ImageEnhance.Contrast(img)
        bright_enhancer = ImageEnhance.Brightness(img)
        img_contrast_en = contrast_enhancer.enhance(factor)
        aug_cr_filename = filename.replace('.png', '') + '_cr_' + str(factor) + '.png'
        aug_br_filename = filename.replace('.png', '') + '_br_' + str(factor) + '.png'
        img_contrast_en.save(dir_path + '/' + aug_cr_filename)
        img_bright_en = bright_enhancer.enhance(factor)
        img_bright_en.save(dir_path + '/' + aug_br_filename)
        train_set_aug_arr.append(
            {'Image_path': 'augmented_train_set/' + aug_cr_filename, 'Defect': row['Defect']})
        train_set_aug_arr.append(
            {'Image_path': 'augmented_train_set/' + aug_br_filename, 'Defect': row['Defect']})
    return train_set_aug_arr


def evaluate_aug_test_set(aug_csv, image_dir, model_preprocessing_func, cnn_model, batch_size, factor, aug_type):
    aug_test_df = pd.read_csv(aug_csv, sep=",")
    aug_test_datagen = ImageDataGenerator(preprocessing_function=model_preprocessing_func)
    aug_test_generator = aug_test_datagen.flow_from_dataframe(
        aug_test_df,
        directory=image_dir,
        target_size=(128, 128),
        x_col='Image',
        y_col='Defect',
        batch_size=batch_size,
        shuffle=False,
        class_mode='raw')
    test_errors = cnn_model.evaluate(aug_test_generator)  # steps =
    print("MSE Score on %s augmented test-set of factor %.1f is %.4f" % (aug_type, factor, test_errors[1]))


def print_aug_test_set_metrics(images_path, preprocessing_func, cnn_model, batch_size):
    for factor in [0.8, 0.9, 1.1, 1.2]:
        aug_br_csv_file_path = images_path + '/augmented_test_set_new/overview_test_aug_br_' + str(
            factor) + '.csv'
        aug_cr_csv_file_path = images_path + '/augmented_test_set_new/overview_test_aug_cr_' + str(
            factor) + '.csv'
        br_image_dir = images_path + '/augmented_test_set_new/brightness_' + str(factor)
        cr_image_dir = images_path + '/augmented_test_set_new/contrast_' + str(factor)
        print("---Brightness= " + str(factor) + "--------------")
        evaluate_aug_test_set(aug_br_csv_file_path, br_image_dir, preprocessing_func, cnn_model,
                              batch_size, factor, 'brightness')
        print("---Contrast= " + str(factor) + "--")
        evaluate_aug_test_set(aug_cr_csv_file_path, cr_image_dir, preprocessing_func, cnn_model,
                              batch_size, factor, 'contrast')
