import * as tf from "@tensorflow/tfjs";

async function getImageFromURL(url: string): Promise<HTMLImageElement> {
    const image = new Image();
    image.src = url + `?timestamp=${Date.now()}`;
    image.crossOrigin = "anonymous";
    await image.decode();
    return image;
}

async function getTensor() {
    const url = `https://deepmars.s3.amazonaws.com/dog/dog.jpg`;
    const image = await getImageFromURL(url);
    return tf.browser.fromPixels(image);
}

async function* dataGenerator() {
    const tensor = await getTensor();
    for (let i = 0; i < 10; i++) yield tensor;
}

export async function getTFDataset() {
    const ds = tf.data.generator(dataGenerator as any);
    await ds.forEachAsync(e => console.log(e));
}


