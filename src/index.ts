import * as tf from '@tensorflow/tfjs';
import { windowReady } from "./ui";

import { buildSequentialModel } from "./model";

export function printInfo() {
    console.log(`Current version ${tf.version.tfjs}`);
}

let current_model = [
    tf.layers.conv2d({ inputShape: [100, 100, 3], filters: 12, kernelSize: 1, activation: "relu" }),
    tf.layers.conv2d({ filters: 12, kernelSize: 1, activation: "relu" }),
    tf.layers.conv2d({ filters: 12, kernelSize: 1, activation: "softmax" })
]

class Table {

    private element: HTMLTableElement = document.createElement("table");

    constructor(rows: number, cols: number, titles: string[], data: string[]) {
        const header = this.element.createTHead();
        header.insertRow();
        for (let i = 0; i < cols; i++) {
            header.rows[0].insertCell();
            header.rows[0].cells[i].innerText = titles[i];
        }
        const body = this.element.createTBody();
        for (let i = 0; i < rows; i++) {
            body.insertRow();
            for (let j = 0; j < cols; j++) {
                body.rows[i].insertCell();
                body.rows[i].cells[j].innerText = data[(i * cols) + j];
            }
        }
    }
    getElement() { return this.element; }

}

async function main() {

    await windowReady();

    const app = document.getElementById("app");

    const summary_table = new Table(2, 3, ["Layer", "Input Size", "w"], ["hi", "2", "2", "1", "2", "w"]);

    app.append(summary_table.getElement());


}

main();

