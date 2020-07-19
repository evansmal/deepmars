import * as tf from '@tensorflow/tfjs';
import * as tfjs from "@tensorflow/tfjs-vis";

import * as React from "react";

import { DatasetLoader } from "../dataset";
import { TerrainClass } from "../types";

interface SinglePredictionProps {
    true_class: number;
    predicted_class: number[];
    image: HTMLImageElement;
}


export const SinglePrediction = (props: SinglePredictionProps) => {

    const capitalize = (message: string) => {
        return message.charAt(0).toUpperCase() + message.slice(1);
    }

    const getLabelName = (index: number) => {
        return capitalize(TerrainClass[index]);
    }

    const argMax = (a: number[]) => {
        return a.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
    }

    const getClassName = (label: number, prediction: number[]) => {
        if (argMax(prediction) === label) {
            return "prediction-img-right"
        } else {
            return "prediction-img-wrong"
        }

    }

    return (
        <div className="prediction-container">
            <div className="prediction-truth">
                <img className={getClassName(props.true_class, props.predicted_class)} src={props.image.src} />
                <p> {getLabelName(props.true_class)} </p>
            </div>
            <div className="prediction-progress">
                <label>Sand </label>
                <progress className="prediction-label" value={props.predicted_class[0]} max="1"></progress>
                <br />

                <label>Rocks </label>
                <progress className="prediction-label" value={props.predicted_class[1]} max="1"></progress>
                <br />

                <label>Bedrock </label>
                <progress className="prediction-label" value={props.predicted_class[2]} max="1"></progress>
                <br />
            </div>
        </div>
    )
}

interface PredictionResultsProps {
    model: tf.Sequential;
    dataset: DatasetLoader;
}

export const PredictionResults = (props: PredictionResultsProps) => {

    const [datasetReady, setDatasetReady] = React.useState(false);
    const [imagesToRender, setImagesToRender] = React.useState([]);

    React.useEffect(() => {
        const interval = setInterval(() => {
            if (props.model !== null && datasetReady === true) {
                handleClick();
            }
        }, 1250);
        return () => clearInterval(interval);
    }, [props.model, datasetReady]);

    props.dataset.onDownloadComplete(() => {
        setDatasetReady(true);
    });

    const getStatus = () => {
        if (props.model == null) {
            return "Waiting for a model...";
        } else if (datasetReady == false) {
            return "Waiting for the dataset to be ready...";
        } else {
            return "Running predictions";
        }
    }

    const handleClick = () => {
        if (datasetReady) {
            const images: JSX.Element[] = [];
            const result: number[][] = props.model.predict(props.dataset.testing_dataset.x, { batchSize: 4 }).arraySync()
            const labels: number[] = props.dataset.testing_dataset.y.argMax(-1).arraySync() as number[];
            for (let i = 0; i < labels.length; i++) {
                images.push(<SinglePrediction key={i} true_class={labels[i]} predicted_class={result[i]} image={props.dataset.testing_dataset.images[i]} />);
            }
            setImagesToRender(images);
        }
    }

    return (
        <div>
            <h2>Test Results</h2>
            <p>Status: {getStatus()}</p>
            <div className="prediction-all">
                <div className="prediction-grid">
                    {imagesToRender}
                </div>
            </div>
        </div>
    )
}
