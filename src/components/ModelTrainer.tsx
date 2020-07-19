import * as tf from '@tensorflow/tfjs';
import * as tfjs from "@tensorflow/tfjs-vis";

import * as React from "react";

import { DatasetLoader } from "../dataset";

export interface ModelTrainerProps {
    model: tf.Sequential;
    dataset: DatasetLoader;
}

export const ModelTrainer = (props: ModelTrainerProps) => {

    const [optimizer, setOptimizer] = React.useState("adam");
    const [learningRate, setLearningRate] = React.useState(0.001);
    const [batchSize, setBatchSize] = React.useState(16);
    const [epochs, setEpochs] = React.useState(10);
    const [datasetReady, setDatasetReady] = React.useState(false);

    props.dataset.onDownloadComplete(() => {
        console.log("Dataset is ready!");
        setDatasetReady(true);

    });

    const getStatus = (model: tf.Sequential | null) => {
        if (model !== null && datasetReady == true) return "Ready";
        else if (model !== null && datasetReady == false) return "Waiting for dataset to download...";
        else return "Waiting for a model...";
    }

    const isTrainingDisabled = (model: tf.Sequential | null) => {
        if (model !== null && datasetReady == true) return false;
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

    const onSelectEpochs = (event) => {
        setEpochs(parseInt(event.target.value));
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
            props.model.compile({ loss: "categoricalCrossentropy", optimizer: opt, metrics: ["accuracy"] });
        } else {
            console.error("Model is `null`");
        }
        const metrics = ['loss', 'acc', "val_loss", "val_acc"];
        const container = {
            name: 'Model Training', tab: "Training Performance", styles: { height: '1000px' }
        };
        const fitCallbacks = tfjs.show.fitCallbacks(container, metrics);

        props.model.fit(props.dataset.training_dataset.x, props.dataset.training_dataset.y, { validationSplit: 0.3, shuffle: true, callbacks: fitCallbacks, batchSize: batchSize, epochs: epochs });
    }

    const evalModel = () => {

        const res = props.model.predict(props.dataset.testing_dataset.x).argMax(-1);
        const labels = props.dataset.testing_dataset.y.argMax(-1);

        const conf_container = { name: 'Confusion Matrix', tab: 'Evaluation' };
        tfjs.metrics.confusionMatrix(labels, res).then(matrix => {
            tfjs.render.confusionMatrix(
                conf_container, { values: matrix, tickLabels: ["Sand", "Rocks", "Bedrock"] });
        });

        const acc_container = { name: 'Accuracy', tab: 'Evaluation' };
        tfjs.metrics.perClassAccuracy(labels, res).then((acc) => {
            tfjs.show.perClassAccuracy(acc_container, acc, ["Sand", "Rocks", "Bedrock"]);
        });


    }

    return (
        <div>
            <h2> Model Trainer </h2>
            <p>Choose an optimization algorithm and tune training hyperparameters.</p>
            <p>Status: {getStatus(props.model)} </p>

            <label>Optimizer: </label>
            <select id="optimizers" value={optimizer} onChange={onSelectOptimizers}>
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
            </select>

            <br />
            <br />
            <label htmlFor="epochs">Epochs: </label>
            <input id="epochs" onChange={onSelectEpochs} value={epochs} min="1" type="number" />

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

            <button disabled={isTrainingDisabled(props.model)} onClick={compileModel}>Run Training</button>

            <br />

            <button disabled={isTrainingDisabled(props.model)} onClick={evalModel}>Evaluate Performance</button>

        </div>
    )
}
