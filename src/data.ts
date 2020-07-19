import * as tf from "@tensorflow/tfjs";

export async function convertToImageElement(buffer: ArrayBuffer): Promise<HTMLImageElement> {
    const blob = new Blob([buffer]);
    const image = new Image();
    image.src = URL.createObjectURL(blob);
    await image.decode();
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

export function convertImageToTensor(examples: HTMLImageElement[]): tf.Tensor {
    const tensors = examples.map((im: HTMLImageElement) => { return tf.browser.fromPixels(im, 1) });
    const resize = tensors.map(t => { return tf.image.resizeBilinear(t, [50, 50], true) })
    const reshape = resize.map(t => { return t.reshape([1, 50, 50, 1]) })
    const norm = reshape.map(t => { return normalizeTensor(t) })
    return tf.concat(norm, 0);
}

export function convertToOneHot(labels: number[]) {
    const one_hot = labels.map(i => { return tf.oneHot(i, 3).expandDims(0) })
    return tf.concat(one_hot, 0);
}
