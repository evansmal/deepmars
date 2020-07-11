import * as tf from '@tensorflow/tfjs';

export function buildSequentialModel(layers: tf.layers.Layer[]) {
    return tf.sequential({ layers: layers });
}
