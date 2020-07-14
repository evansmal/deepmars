import * as tf from '@tensorflow/tfjs';

import { h, FunctionalComponent } from "preact";
import { useState } from "preact/hooks";

export interface ModelTrainerProps {
    model: tf.Sequential;
}

export const ModelTrainer = (props: ModelTrainerProps) => {

    const [optimizer, setOptimizer] = useState("adam");
    const [learningRate, setLearningRate] = useState(0.001);
    const [batchSize, setBatchSize] = useState(4);

    const getStatus = (model: tf.Sequential | null) => {
        if (model !== null) return "Ready";
        else return "Not Ready";
    }

    const isTrainingDisabled = (model: tf.Sequential | null) => {
        if (model !== null) return false;
        else return true;
    }

    const onSelectLearningRate = (event) => {
        setLearningRate(parseFloat(event.target.value));
    }

    const onSelectBatchSize = (event) => {
        setBatchSize(parseInt(event.target.value));
    }

    const onSelectOptimizers = (event) => {
        setOptimizer(event.target.value);
    }

    const compileModel = () => {
        console.log("Starting training!");

        let opt = null;
        if (optimizer == "adam") {
            opt = tf.train.adam(learningRate);
        } else {
            opt = tf.train.sgd(learningRate);
        }

        if (props.model) {
            props.model.compile({ loss: "categoricalCrossentropy", optimizer: opt });
        } else {
            console.error("Model is `null`");
        }


    }

    return (
        <div>
            <h2> Model Trainer </h2>
            <p>Status: {getStatus(props.model)} </p>

            <h3>Training Hyperparameters</h3>

            <label>Optimizer: </label>
            <select id="optimizers" value={optimizer} onChange={onSelectOptimizers}>
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
            </select>

            <br />
            <br />

            <label htmlFor="learning_rate">Learning rate: </label>
            <input id="learning_rate" onChange={onSelectLearningRate} step="0.001" value={learningRate} min="0" type="number" />

            <br />
            <br />

            <label htmlFor="batch_size">Batch size: </label>
            <input id="batch_size" onChange={onSelectBatchSize} value={batchSize} step="1" max="100" min="1" type="number" />

            <br />
            <br />

            <button disabled={isTrainingDisabled(props.model)} onClick={compileModel}>Compile and Start Training</button>

        </div>
    )
}
