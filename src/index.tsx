import * as tf from '@tensorflow/tfjs';

import * as ReactDOM from "react-dom";
import * as React from "react";

import { Provider } from "react-redux";
import { createStore } from "redux";

import { NetworkBuilder } from "./ui/builder";
import { ModelTrainer } from "./ui/trainer";

export function printInfo() {
    console.log(`Current version ${tf.version.tfjs}`);
}

const root_reducer = () => {
    time: Date.now()
}

const store = createStore(root_reducer);

const App = () => {
    const [currentNetwork, setCurrentNetwork] = React.useState(null);
    const onSubmitNetwork = (network: tf.Sequential) => {
        console.log("Submitted", network);
        setCurrentNetwork(network);
    }
    return (
        <div>
            <Provider store={store}>
                <NetworkBuilder onSubmitNetwork={onSubmitNetwork} />
                <br />
                <ModelTrainer model={currentNetwork} />
            </Provider>
        </div>
    );
}

const app = document.createElement("div");
document.body.appendChild(app);
ReactDOM.render(<App />, app);
