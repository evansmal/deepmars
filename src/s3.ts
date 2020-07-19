import { S3 } from "aws-sdk";

import { FullDataset, RawPair, TerrainClass } from "./types";

const S3_TRAIN_PREFIXES = ["train/sand", "train/bedrock", "train/rocks"];
const S3_TEST_PREFIXES = ["test/sand", "test/bedrock", "test/rocks"];

export async function listFiles(bucket_name: string, prefix: string): Promise<Array<string>> {
    return new Promise((resolve, reject) => {
        const s3 = new S3({ params: { Bucket: bucket_name, Prefix: prefix } });
        s3.makeUnauthenticatedRequest("listObjects", (err: any, data: any) => {
            if (err) reject();
            const content = data.Contents as Array<any>;
            resolve(content.map((item) => {
                return item.Key;
            }));
        });
    });
}

export async function listFilesConcurrent(bucket_name: string, prefixes: string[]): Promise<Array<string>> {
    const res = await Promise.all(prefixes.map((pref) => {
        return listFiles(bucket_name, pref);
    }));
    return [].concat(...res);
}

export async function getTrainingSetURLs(bucket_name: string): Promise<string[]> {
    return await listFilesConcurrent(bucket_name, [...S3_TRAIN_PREFIXES]);
}

export async function getTestingSetURLs(bucket_name: string): Promise<string[]> {
    return await listFilesConcurrent(bucket_name, [...S3_TEST_PREFIXES]);
}

export async function getDatasetURLs(bucket_name: string): Promise<string[]> {
    return await listFilesConcurrent(bucket_name, [...S3_TRAIN_PREFIXES, ...S3_TEST_PREFIXES]);
}

export async function getFile(bucket_name: string, key: string): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
        const s3 = new S3({ params: { Bucket: bucket_name, Key: key } });
        s3.makeUnauthenticatedRequest("getObject", (err: any, data: any) => {
            if (err) reject();
            resolve(data.Body)
        });
    });
}

export async function getMultipleFile(bucket_name: string, keys: string[]): Promise<ArrayBuffer[]> {
    return Promise.all(keys.map((key) => {
        return getFile(bucket_name, key);
    }));
}

export async function storeMultipleFileInStorage(bucket_name: string, keys: string[]): Promise<ArrayBuffer[]> {
    return Promise.all(keys.map((key) => {
        return getFile(bucket_name, key);
    }));
}

function getClassnames(paths: string[]): TerrainClass[] {
    return paths.map(path => { return TerrainClass[path.split("/")[1]] });
}

export async function getFullDatasetFromS3(bucket_name: string): Promise<FullDataset<RawPair>> {
    const [train, test] = await Promise.all([getTrainingSetURLs(bucket_name), getTestingSetURLs(bucket_name)]);

    const shuffled_train = train.sort(() => Math.random() - 0.5)
    const shuffled_test = test.sort(() => Math.random() - 0.5)

    const [training_data, test_data] = await Promise.all([getMultipleFile(bucket_name, shuffled_train), getMultipleFile(bucket_name, shuffled_test)]);
    return {
        training: { x: training_data, y: getClassnames(shuffled_train) },
        test: { x: test_data, y: getClassnames(shuffled_test) }
    }
}
