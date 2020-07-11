import * as tf from "@tensorflow/tfjs";
import * as tfjs from "@tensorflow/tfjs-vis";

import { h, FunctionalComponent } from "preact";
import { useState } from "preact/hooks";

import { buildSequentialModel } from "./model";

interface DenseBuilderProps {
    onAddLayer: (units: number) => void;
}

export const DenseBuilder: FunctionalComponent<DenseBuilderProps> = (props) => {

    const [selectedUnits, setSelectedUnits] = useState(64);

    const onSubmit = () => {
        props.onAddLayer(selectedUnits);
    }

    const onChangeUnits = (event) => {
        setSelectedUnits(event.target.value);
    }


    return (
        <div>
            <h3>Dense </h3>

            <label htmlFor="units">Number of units</label>
            <input id="units" onChange={onChangeUnits} value={selectedUnits} type="number" />

            <br />

            <button onClick={onSubmit} type="submit">Add</button>
        </div>
    )
}

interface Conv2dBuilderProps {
    onAddLayer: (filters: number, kernelSize: number) => void;
}

export const Conv2dBuilder: FunctionalComponent<Conv2dBuilderProps> = (props) => {

    const [selectedFilters, setSelectedFilters] = useState(12);
    const [selectedKernelSize, setSelectedKernelSize] = useState(3);

    const onSubmit = () => {
        props.onAddLayer(selectedFilters, selectedKernelSize);
    }

    const onChangeFilters = (event) => {
        setSelectedFilters(event.target.value);
    }

    const onChangeKernelSize = (event) => {
        setSelectedKernelSize(event.target.value);
    }

    return (
        <div>
            <h3>Convolution2D </h3>

            <label htmlFor="filters">Number of filters</label>
            <input id="filters" onChange={onChangeFilters} value={selectedFilters} type="number" />

            <br />

            <label htmlFor="kernel">Number of filters</label>
            <input id="kernel" onChange={onChangeKernelSize} value={selectedKernelSize} type="number" />

            <br />

            <button onClick={onSubmit} type="submit">Add</button>
        </div>
    )
}

export interface MaxPoolingBuilderProps {
    onAddLayer: (size: number) => void;
}

export const MaxPoolingBuilder: FunctionalComponent<MaxPoolingBuilderProps> = (props) => {

    const [selectedSize, setSelectedSize] = useState(2);

    const onSubmit = () => {
        props.onAddLayer(selectedSize);
    }

    const onChangeSize = (event) => {
        setSelectedSize(event.target.value);
    }

    return (
        <div>
            <h3>MaxPooling2D</h3>
            <label htmlFor="size">Number of filters</label>
            <input id="size" onChange={onChangeSize} value={selectedSize} type="number" />

            <br />

            <button onClick={onSubmit} type="submit">Add</button>
        </div>
    )
}

interface NetworkConfiguration {


}

export interface NetworkBuilderProps {
    onSubmitNetwork: (config: NetworkConfiguration) => void;
}

export const NetworkBuilder: FunctionalComponent = (props) => {

    const [model, setModel] = useState([]);

    const addConv = (filters: number, kernelSize: number) => {
        if (model.length == 0) {
            setModel([...model, tf.layers.conv2d({ inputShape: [100, 100, 3], filters: filters, kernelSize: kernelSize, activation: "relu" })]);
        } else {
            setModel([...model, tf.layers.conv2d({ filters: filters, kernelSize: kernelSize, activation: "relu" })]);
        }
    }

    const addPool = (size: number) => {
        if (model.length == 0) {
            setModel([...model, tf.layers.maxPool2d({ inputShape: [100, 100, 3], poolSize: size })]);
        } else {
            setModel([...model, tf.layers.maxPool2d({ poolSize: size })]);
        }
    }

    const addDense = (units: number) => {

        if (units < 1) {
            console.error("Number of units must be >0");
            return;
        }
        if (model.length == 0) {
            setModel([...model, tf.layers.flatten({ inputShape: [100, 100, 3] }), tf.layers.dense({ units: units })]);
            return;
        }

        let last_layer = model[model.length - 1];
        if (last_layer.constructor.name == "Dense") {
            setModel([...model, tf.layers.dense({ units: units })]);
        } else {

            setModel([...model, tf.layers.flatten(), tf.layers.dense({ units: units })]);
        }
    }

    const removeLastLayer = () => {
        if (model.length <= 1) setModel([]);
        else setModel([...model.slice(0, model.length - 1)]);
    }

    const buildModel = () => {
        if (model.length == 0) {
            console.error("No layers selected. Doing nothing.");
            return
        }
        const seq = buildSequentialModel(model);

        tfjs.show.modelSummary({ name: "Model Summary", tab: "Model Inspection" }, seq);
    }

    return (
        <div>
            <h2> Architecture Builder </h2>
            <Conv2dBuilder onAddLayer={addConv} />
            <MaxPoolingBuilder onAddLayer={addPool} />
            <DenseBuilder onAddLayer={addDense} />

            <br />

            <p>Number of layers: {model.length}</p>
            <button onClick={buildModel}>Compile Model</button>
            <button onClick={removeLastLayer}>Remove</button>

        </div>
    )

}



