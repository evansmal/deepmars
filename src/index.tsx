import * as tf from '@tensorflow/tfjs';

import * as ReactDOM from "react-dom";
import * as React from "react";

import { NetworkBuilder } from "./components/NetworkBuilder";
import { ModelTrainer } from "./components/ModelTrainer";
import { PredictionResults } from "./components/PredictionResults";
import { ActivationBrowser } from "./components/ActivationBrowser";

import { DatasetLoader } from "./dataset";

import "./components/Styling.css";

const dataset = new DatasetLoader();
dataset.downloadData();

const App = () => {

    const [currentNetwork, setCurrentNetwork] = React.useState(null);
    const [dataDownloaded, setDataDownloaded] = React.useState(false);

    dataset.onDownloadComplete(() => {
        setDataDownloaded(true);
    })

    const getDataStatus = () => {
        if (dataDownloaded === true) {
            return "Dataset download is complete";
        } else {
            return "Dataset is currently being downloaded..."
        }
    }

    const onSubmitNetwork = (network: tf.Sequential) => {
        setCurrentNetwork(network);
    }

    const divStyle = {

    };

    return (
        <div>
            <h3>{getDataStatus()}</h3>
            <hr />

            <div style={divStyle}>
                <NetworkBuilder onSubmitNetwork={onSubmitNetwork} />
            </div>

            <br />
            <hr />

            <div style={divStyle}>
                <ModelTrainer model={currentNetwork} dataset={dataset} />
            </div>

            <br />
            <hr />

            <div style={divStyle}>
                <PredictionResults model={currentNetwork} dataset={dataset} />
            </div>

            <br />
            <hr />
        </div>
    );
}

const app = document.createElement("div");
document.body.appendChild(app);
ReactDOM.render(<App />, app);
