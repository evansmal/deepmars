import * as tf from "@tensorflow/tfjs";

import { getFullDatasetFromS3 } from "./s3";
import { RawPair } from "./types";
import { convertImageToTensor, convertRawExamplesToImage, convertToOneHot } from "./data";

interface TensorDataset {
    x: tf.Tensor;
    y: tf.Tensor;
}

async function downloadAndPreprocess(raw: RawPair): Promise<TensorDataset> {
    const images = await convertRawExamplesToImage(raw.x)
    return { x: convertImageToTensor(images), y: convertToOneHot(raw.y) }
}

export class DatasetLoader {

    public testing_dataset: TensorDataset;
    public training_dataset: TensorDataset;

    private on_ready_callback: () => void;

    async downloadData() {
        console.log("Starting download...");
        const t0 = performance.now()
        const raw = await getFullDatasetFromS3("deepmars");
        const t1 = performance.now()
        console.log(`Took ${t1 - t0} ms to get raw images`);

        this.training_dataset = await downloadAndPreprocess(raw.training);
        const t2 = performance.now();
        console.log(`Took ${t2 - t1} ms to get preprocess training data`);

        this.testing_dataset = await downloadAndPreprocess(raw.test);
        this.on_ready_callback();
        console.log("Done download!");

    }

    onDownloadComplete(callback: () => void) {
        this.on_ready_callback = callback;
    }
}
