{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uLb1_v6plJlC",
        "outputId": "c83dadd1-ef8d-49a7-b60e-4b3068d69b8c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\",force_remount-True)\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.preprocessing.image import ImageDataGenerator"
      ],
      "metadata": {
        "id": "foE-CZQG5XXh"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)\n",
        "test_datagen=ImageDataGenerator(rescale=1./255)"
      ],
      "metadata": {
        "id": "x8mp0JdC6kr4"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Flow training images in batches of 128 using train_datagen generator\n",
        "train_generator = train_datagen.flow_from_directory(\n",
        "        'Data',  # This is the source directory for training images\n",
        "        target_size=(200, 200),  # All images will be resized to 200 x 200\n",
        "        batch_size=5,\n",
        "        # Specify the classes explicitly\n",
        "        classes = ['APPLES','BANANA','ORANGE','PINEAPPLE','WATERMELON'],\n",
        "        # Since we use categorical_crossentropy loss, we need categorical labels\n",
        "        class_mode='categorical')\n",
        "# Flow training images in batches of 128 using test_datagen generator\n",
        "test_generator = test_datagen.flow_from_directory(\n",
        "    'Data', #this sources directory for testing images\n",
        "    target_size=(200, 200), #All images will beresized to 200 x 200\n",
        "    batch_size=5,\n",
        "    # Specify the classes explicitly\n",
        "     classes = ['APPLES','BANANA','ORANGE','PINEAPPLE','WATERMELON'],\n",
        "     # Since we use categorical_crossentropy loss, we need categorical labels\n",
        "      class_mode='categorical')\n",
        "\n"
      ],
      "metadata": {
        "id": "JXDjb7-rAW0c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(train_generator.class_indices)#checking the number of classes \n",
        "{'APPLES':0, 'BANANA':1, 'ORANGE':2,'PINEAPPLE':3,'WATERMELON':4}\n",
        "print(test_generator.class_indices)#checking the number of classes\n",
        "{'APPLES':0, 'BANANA':1, 'ORANGE':2,'PINEAPPLE':3,'WATERMELON':4}\n",
        "from collections import Counter as c\n",
        "c(train_generator.labels)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_mkR_TPaMXhf",
        "outputId": "7391f2e8-2468-47ea-e86e-4da7610a1f0f"
      },
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'APPLES': 0, 'BANANA': 1, 'ORANGE': 2, 'PINEAPPLE': 3, 'WATERMELON': 4}\n",
            "{'APPLES': 0, 'BANANA': 1, 'ORANGE': 2, 'PINEAPPLE': 3, 'WATERMELON': 4}\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Counter()"
            ]
          },
          "metadata": {},
          "execution_count": 50
        }
      ]
    }
  ]
}
