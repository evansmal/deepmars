import * as tf from "@tensorflow/tfjs";

export interface State {
    model: {
        selected_model: tf.Sequential | null
    }
}

export const INITIAL_STATE: State = {
    model: {
        selected_model: null
    }
}
