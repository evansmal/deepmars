import * as tf from '@tensorflow/tfjs';

export enum TerrainClass {
    sand,
    rocks,
    bedrock,
}

export interface FullDataset<T extends ExamplePair> {
    training: T;
    test: T;
}

export interface ExamplePair {
    y: TerrainClass[];
}

export interface RawPair extends ExamplePair {
    x: ArrayBuffer[];
}

export interface ImagePair extends ExamplePair {
    x: HTMLImageElement[];
}

export interface TensorPair extends ExamplePair {
    x: tf.Tensor[];
}
