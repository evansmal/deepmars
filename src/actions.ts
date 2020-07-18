import * as tf from "@tensorflow/tfjs"

export const SET_MODEL = 'SET_MODEL'

interface SetModelAction {
    type: typeof SET_MODEL
    model: tf.Sequential
}

export function setModel(model: tf.Sequential): SetModelAction {
    return {
        type: SET_MODEL,
        model: model
    }
}
