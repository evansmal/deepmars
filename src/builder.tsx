import * as tf from "@tensorflow/tfjs";
import * as tfjs from "@tensorflow/tfjs-vis";

import { h, FunctionalComponent } from "preact";
import { useState } from "preact/hooks";

import { buildSequentialModel, getDefaultModelLayers } from "./model";

interface DenseBuilderProps {
    onAddLayer: (units: number) => void;
}

export const DenseBuilder = (props: DenseBuilderProps) => {

    const [selectedUnits, setSelectedUnits] = useState(64);

    const onSubmit = () => {
        props.onAddLayer(parseInt(selectedUnits));
    }

    const onChangeUnits = (event) => {
        setSelectedUnits(event.target.value);
    }


    return (
        <div>
            <h3>Dense </h3>

            <label htmlFor="units">Number of units: </label>
            <input id="units" onChange={onChangeUnits} value={selectedUnits} min="1" type="number" />

            <br />

            <button onClick={onSubmit} type="submit">Add</button>
        </div>
    )
}

interface Conv2dBuilderProps {
    onAddLayer: (filters: number, kernelSize: number) => void;
}

export const Conv2dBuilder = (props: Conv2dBuilderProps) => {

    const [selectedFilters, setSelectedFilters] = useState(12);
    const [selectedKernelSize, setSelectedKernelSize] = useState(3);

    const onSubmit = () => {
        props.onAddLayer(parseInt(selectedFilters), parseInt(selectedKernelSize));
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

            <label htmlFor="filters">Number of filters: </label>
            <input id="filters" onChange={onChangeFilters} value={selectedFilters} min="1" type="number" />

            <br />

            <label htmlFor="kernel">Kernel size: </label>
            <input id="kernel" onChange={onChangeKernelSize} value={selectedKernelSize} min="1" type="number" />

            <br />

            <button onClick={onSubmit} type="submit">Add</button>
        </div>
    )
}

export interface MaxPoolingBuilderProps {
    onAddLayer: (size: number) => void;
}

export const MaxPoolingBuilder = (props: MaxPoolingBuilderProps) => {

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
            <label htmlFor="size">Pooling size: </label>
            <input id="size" onChange={onChangeSize} value={selectedSize} min="1" type="number" />

            <br />

            <button onClick={onSubmit} type="submit">Add</button>
        </div>
    )
}

export interface NetworkBuilderProps {
    onSubmitNetwork: (model: tf.Sequential) => void;
}

export const NetworkBuilder = (props: NetworkBuilderProps) => {

    const onSubmitNetwork = props.onSubmitNetwork;

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
        tfjs.show.modelSummary({ name: "Model Summary", tab: "Architecture Builder" }, seq);
        onSubmitNetwork(seq);
    }

    const loadDefaultNetwork = () => {
        setModel(getDefaultModelLayers([100, 100, 1], 9));
    }

    return (
        <div>
            <h2> Architecture Builder </h2>
            <Conv2dBuilder onAddLayer={addConv} />
            <MaxPoolingBuilder onAddLayer={addPool} />
            <DenseBuilder onAddLayer={addDense} />

            <br />

            <button onClick={loadDefaultNetwork}>Use Default Architecture</button>

            <br />

            <p>Number of layers: {model.length}</p>
            <button onClick={buildModel}>Confirm</button>
            <button onClick={removeLastLayer}>Remove</button>


        </div>
    )

}
