import * as tf from "@tensorflow/tfjs";

export interface SetSelectedModel {
    type: "SET_SELECTED_MODEL";
    model: tf.Sequential;
}

export function setSelectedModel(model: tf.Sequential): SetSelectedModel {
    return { type: "SET_SELECTED_MODEL", model: model };
}

export type Action = SetSelectedModel;
