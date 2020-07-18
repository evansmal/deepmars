import * as tf from '@tensorflow/tfjs';
import * as tfjs from "@tensorflow/tfjs-vis";

import { h } from "preact";
import { useState, useEffect } from "preact/hooks";

import { getFullDatasetFromS3 } from "../s3";
import { FullDataset, TensorPair } from "../types";
import { convertImageToTensor, convertRawExamplesToImage, convertToOneHot } from "../data";

export interface ModelTrainerProps {
    model: tf.Sequential;
}

export const ModelTrainer = (props: ModelTrainerProps) => {

    const [optimizer, setOptimizer] = useState("adam");
    const [learningRate, setLearningRate] = useState(0.001);
    const [batchSize, setBatchSize] = useState(32);
    const [trainingData, setTrainingData] = useState(null);
    const [testingData, setTestingData] = useState(null);

    useEffect(() => {
        const bucket_name = "deepmars"
        getFullDatasetFromS3(bucket_name).then((raw) => {
            convertRawExamplesToImage(raw.training.x).then((data) => {
                const training = { x: convertImageToTensor(data), y: convertToOneHot(raw.training.y) }
                setTrainingData(training);
            });
            convertRawExamplesToImage(raw.test.x).then((data) => {
                const testing = { x: convertImageToTensor(data), y: convertToOneHot(raw.test.y) }
                setTestingData(testing);
            });
        });
        return () => {
        };
    }, []);

    const getStatus = (model: tf.Sequential | null) => {
        if (model !== null && trainingData !== null) return "Ready";
        else if (model !== null && trainingData == null) return "Loading dataset - please wait...";
        else return "Not Ready";
    }

    const isTrainingDisabled = (model: tf.Sequential | null) => {
        if (model !== null && trainingData !== null) return false;
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
        const metrics = ['loss', 'acc'];
        const container = {
            name: 'Model Training', tab: "Model Trainer", styles: { height: '1000px' }
        };
        const fitCallbacks = tfjs.show.fitCallbacks(container, metrics);
        props.model.fit(trainingData.x, trainingData.y, { validationSplit: 0.2, shuffle: true, callbacks: fitCallbacks, batchSize: batchSize, epochs: 50 });
    }

    const evalModel = () => {
        console.log("Evaluating");
        const res = props.model.predict(testingData.x, testingData.y).argMax(-1);
        const labels = testingData.y.argMax(-1);
        console.log(res, labels)
        res.print(true)
        const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
        tfjs.metrics.confusionMatrix(labels, res).then(matrix => {
            tfjs.render.confusionMatrix(
                container, { values: matrix, tickLabels: ["Sand", "Bedrock", "Rocks"] });
        });

    }

    return (
        <div>
            <h2> Model Trainer </h2>
            <p>Status: {getStatus(props.model)} </p>

            <br />

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

            <button disabled={isTrainingDisabled(props.model)} onClick={compileModel}>Start Training</button>
            <button disabled={isTrainingDisabled(props.model)} onClick={evalModel}>Evaluate</button>

        </div>
    )
}
