import * as tf from '@tensorflow/tfjs';

import { h, render } from "preact";
import { useState } from "preact/hooks";

import { NetworkBuilder } from "./builder";
import { ModelTrainer } from "./trainer";

export function printInfo() {
    console.log(`Current version ${tf.version.tfjs}`);
}

const App = () => {

    const [currentNetwork, setCurrentNetwork] = useState(null);

    const onSubmitNetwork = (network: tf.Sequential) {
        console.log("Submitted", network);
        setCurrentNetwork(network);
    }

    return (
        <div>
            <NetworkBuilder onSubmitNetwork={onSubmitNetwork} />
            <br />
            <ModelTrainer model={currentNetwork} />
        </div>
    );
}

render(<App />, document.body);

