import * as tf from "@tensorflow/tfjs";

async function promisifyLoad(image: HTMLImageElement, url: string) {
    new Promise((resolve) => {
        image.addEventListener("load", () => {
            resolve(image);
        });
        image.src = url;
    });
}

export async function convertToImageElement(buffer: ArrayBuffer): Promise<HTMLImageElement> {
    const blob = new Blob([buffer]);
    const image = new Image();
    const url = URL.createObjectURL(blob);
    image.src = url;
    await image.decode().catch(err => console.log(err));
    return image
}

export async function getImageFromURL(url: string): Promise<HTMLImageElement> {
    const image = new Image();
    image.src = url + `?timestamp=${Date.now()}`;
    image.crossOrigin = "anonymous";
    await image.decode();
    return image;
}

export async function convertRawExamplesToImage(buffers: ArrayBuffer[]): Promise<HTMLImageElement[]> {
    return Promise.all(buffers.map((b: ArrayBuffer) => {
        return convertToImageElement(b);
    }));
}

function normalizeTensor(tensor: tf.Tensor): tf.Tensor {
    const inputMax = tensor.max();
    const inputMin = tensor.min();
    return tensor.sub(inputMin).div(inputMax.sub(inputMin));
}

function sleep(ms: number) { return new Promise(resolve => setTimeout(resolve, ms)); }

export async function convertImageToTensor(examples: HTMLImageElement[]): Promise<tf.Tensor> {
    const result = []
    for (let example of examples) {
        const tensor = tf.browser.fromPixels(example, 1);
        await sleep(0);
        const resize = tf.image.resizeBilinear(tensor, [50, 50], true).reshape([1, 50, 50, 1]);
        await sleep(0);
        result.push(normalizeTensor(resize));
        await sleep(0);
    }
    return tf.concat(result, 0);
}

export function convertToOneHot(labels: number[]) {
    const one_hot = labels.map(i => { return tf.oneHot(i, 3).expandDims(0) })
    return tf.concat(one_hot, 0);
}
