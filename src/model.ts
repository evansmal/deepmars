import * as tf from '@tensorflow/tfjs';

export function buildSequentialModel(layers: tf.layers.Layer[]) {
    return tf.sequential({ layers: layers });
}

export function getDefaultModelLayers(input_shape: number[], output_classes: number) {

    const IMAGE_WIDTH = input_shape[0];
    const IMAGE_HEIGHT = input_shape[1];
    const IMAGE_CHANNELS = input_shape[2];

    const model: tf.layers.Layer[] = [];

    model.push(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.push(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.push(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.push(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.push(tf.layers.conv2d({
        kernelSize: 3,
        filters: 20,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.push(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.push(tf.layers.flatten());

    const NUM_OUTPUT_CLASSES = output_classes;
    model.push(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    return model;
}
