import * as tf from '@tensorflow/tfjs';
import { h, render, Component } from "preact";

import { windowReady } from "./ui";
import { NetworkBuilder } from "./builder";

export function printInfo() {
    console.log(`Current version ${tf.version.tfjs}`);
}

class App extends Component {
    render() {
        return (
            <div>
                <NetworkBuilder />
            </div>
        );
    }
}

render(<App />, document.body);

async function main() {
    await windowReady();
    printInfo();
}

main();

