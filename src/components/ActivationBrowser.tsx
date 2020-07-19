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

    const canvasRef = React.useRef(null)

    props.dataset.onDownloadComplete(() => {
        setDatasetReady(true);
        console.log(props.dataset.testing_dataset.images[0])
    });

    const computeActivation = () => {

        const new_model = tf.model({ inputs: props.model.inputs, outputs: props.model.layers[4].output });
        const activation = new_model.predict(props.dataset.testing_dataset.x);
        const single_activation = tf.split(activation, 180, 0)[0];
        console.log(single_activation)
        const splt = single_activation.squeeze().split(32, 2);
        splt.forEach(s => {
            tf.browser.toPixels(s, canvasRef.current).then(r => {
                console.log(r);
            });
        })
    }

    const getImages = () => {
        if (datasetReady) {
            const res = []

            computeActivation();
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
            <canvas ref={canvasRef} width={50} height={50} />
            {...getImages()}
        </div>
    )
}
