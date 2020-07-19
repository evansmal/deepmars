import * as tf from '@tensorflow/tfjs';
import * as tfjs from "@tensorflow/tfjs-vis";

import * as React from "react";

import { DatasetLoader } from "../dataset";

interface ActivationBrowserProps {
    model: tf.Sequential;
    dataset: DatasetLoader;
}

export const ActivationBrowser = (props: ActivationBrowserProps) => {

    const [datasetReady, setDatasetReady] = React.useState(false);

    props.dataset.onDownloadComplete(() => {
        setDatasetReady(true);
        console.log(props.dataset.testing_dataset.images[0])
    });

    const getImages = () => {
        if (datasetReady) {
            const res = []
            for (let i = 0; i < 12; i++) {
                res.push(
                    <img key={i} src={props.dataset.testing_dataset.images[i].src} />
                )
            }
            return res;
        } else {
            return [<p key={0}>Waiting for dataset download...</p>]
        }
    }

    return (
        <div>
            <h3>Activation Browser </h3>
            {...getImages()}
        </div>
    )
}
