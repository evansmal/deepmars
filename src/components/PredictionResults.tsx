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
    return (
        <div className="prediction-container">
            <div className="prediction-truth">
                <img className="prediction-img" src={props.image.src} />
                <p> {TerrainClass[props.true_class].toUpperCase()} </p>
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
            console.log('This will run every second!', props.model);
            if (props.model !== null && datasetReady === true) {
                handleClick();
            }
        }, 1000);
        return () => clearInterval(interval);
    }, [props.model, datasetReady]);

    props.dataset.onDownloadComplete(() => {
        setDatasetReady(true);
    });

    const handleClick = () => {
        console.log(props.model)
        if (datasetReady) {
            const images: JSX.Element[] = [];
            const result: number[][] = props.model.predict(props.dataset.testing_dataset.x, { batchSize: 4 }).arraySync()
            const labels: number[] = props.dataset.testing_dataset.y.argMax(-1).arraySync();
            for (let i = 0; i < labels.length; i++) {
                images.push(<SinglePrediction key={i} true_class={labels[i]} predicted_class={result[i]} image={props.dataset.testing_dataset.images[i]} />);
            }
            setImagesToRender(images);
        }
    }

    return (
        <div>
            <h2>Test Results</h2>
            <button onClick={handleClick}>Get Predictions</button>
            <div className="prediction-grid">
                {imagesToRender}
            </div>
        </div>
    )
}
