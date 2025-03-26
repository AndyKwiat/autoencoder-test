import * as tf from '@tensorflow/tfjs';
import { kmeans } from 'ml-kmeans';
import * as clustering from 'density-clustering';

// Simulated dataset: [price, session_count, avg_duration, time_of_day]
const rawData = generateParkingData();

// Normalize the data (simple min-max scaling)
function normalize(data: number[][]): number[][] {
    const transposed = data[0].map((_, colIndex) => data.map(row => row[colIndex]));
    const mins = transposed.map(col => Math.min(...col));
    const maxs = transposed.map(col => Math.max(...col));
    return data.map(row =>
        row.map((value, i) => (value - mins[i]) / (maxs[i] - mins[i]))
    );
}

const normalizedData = normalize(rawData);
const inputTensor = tf.tensor2d(normalizedData);

// Build a simple autoencoder
const encoderDim = 2;
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [4], units: 3, activation: 'relu' }));
model.add(tf.layers.dense({ units: encoderDim, activation: 'relu', name: 'bottleneck' }));
model.add(tf.layers.dense({ units: 3, activation: 'relu' }));
model.add(tf.layers.dense({ units: 4, activation: 'sigmoid' })); // output shape = input shape

model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

await model.fit(inputTensor, inputTensor, {
    epochs: 100,
    verbose: 0,
});

// Extract encoder output (the compressed latent space)
const bottleneckLayer = model.getLayer('bottleneck') as tf.layers.Layer;
const encoder = tf.model({ inputs: model.inputs, outputs: bottleneckLayer.output });

const compressed = encoder.predict(inputTensor) as tf.Tensor;
const compressedArray = await compressed.array() as number[][];

console.log("Compressed points:", compressedArray);
// Run K-Means on the latent vectors
const numClusters = 4;
const kmeansResult = kmeans(compressedArray, numClusters, {});

// Summarize each cluster
for (let i = 0; i < numClusters; i++) {
    const members = rawData.filter((_, idx) => kmeansResult.clusters[idx] === i);
    const count = members.length;
    const mean = members.reduce((sum, row) => {
        return sum.map((val, idx) => val + row[idx]);
    }, [0, 0, 0, 0]).map(val => val / count);

    console.log(`\nCluster ${i}`);
    console.log(`Count: ${count}`);
    console.log(`Avg Price: $${mean[0].toFixed(2)}`);
    console.log(`Avg Sessions: ${mean[1].toFixed(0)}`);
    console.log(`Avg Duration: ${mean[2].toFixed(0)} mins`);
    console.log(`Avg Time of Day: ${mean[3].toFixed(2)}h`);
}

console.log("DB SCANNING");


// DBSCAN parameters:
const eps = 0.2;     // max distance between points in a cluster
const minPts = 2;    // minimum number of points to form a cluster

const dbscan = new clustering.DBSCAN();
const clusters = dbscan.run(compressedArray, eps, minPts);
const noise = dbscan.noise;

// clusters: array of arrays of indexes
console.log("Clusters:", clusters);
console.log("Noise (unclustered points):", noise);

summarizeClusters(rawData, clusters);

function generateParkingData(): number[][] {
    const data: number[][] = [];

    const rand = (min: number, max: number) => Math.random() * (max - min) + min;

    for (let i = 0; i < 50; i++) {
        let price: number;
        let sessionCount: number;
        let avgDuration: number;
        let timeOfDay: number;

        const behaviorType = Math.random();

        if (behaviorType < 0.4) {
            // Morning rush (7–9 AM)
            timeOfDay = rand(7, 9);
            price = rand(2.5, 4.0);
            sessionCount = rand(35, 55);
            avgDuration = rand(15, 35);
        } else if (behaviorType < 0.75) {
            // Afternoon lull (1–3 PM)
            timeOfDay = rand(13, 15);
            price = rand(1.5, 2.5);
            sessionCount = rand(15, 30);
            avgDuration = rand(45, 75);
        } else {
            // Night quiet zone (9 PM – 12 AM)
            timeOfDay = rand(21, 24);
            price = rand(0.0, 1.0);
            sessionCount = rand(0, 10);
            avgDuration = rand(20, 60);
        }

        data.push([price, sessionCount, avgDuration, timeOfDay]);
    }

    return data;
}

function summarizeClusters(rawData: number[][], clusters: number[][]) {
    clusters.forEach((cluster, i) => {
        const members = cluster.map(idx => rawData[idx]);
        const count = members.length;
        const mean = members.reduce((sum, row) =>
            sum.map((val, idx) => val + row[idx]), [0, 0, 0, 0]
        ).map(val => val / count);

        console.log(`\nCluster ${i}`);
        console.log(`Count: ${count}`);
        console.log(`Avg Price: $${mean[0].toFixed(2)}`);
        console.log(`Avg Sessions: ${mean[1].toFixed(0)}`);
        console.log(`Avg Duration: ${mean[2].toFixed(0)} mins`);
        console.log(`Avg Time of Day: ${mean[3].toFixed(2)}h`);
    });
}

