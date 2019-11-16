const tf = require('@tensorflow/tfjs');

import {
  CANW,
  generateData,
  predict,
  drawPoints,
  drawLine
} from './utils';

/**
 * 
 * @param {*} inputA 拟合直线的a值
 * @param {*} inputB 拟合直线的b值
 * @param {*} randomNumberCount 随机数个数
 * @param {*} numIterations 迭代次数
 * @param {*} ctx 画布的2d context
 */
export default async function run(inputA, inputB, randomNumberCount, numIterations, sigma, ctx) {
  ctx.clearRect(0, 0, CANW, CANW);

  const trueCoefficients = {
    a: inputA, b: inputB
  };
  const trainingData = generateData(randomNumberCount, trueCoefficients, false, sigma);

  // 画训练数据点
  await drawPoints(ctx, trainingData.xs, trainingData.ys);

  const [a, b] = tf.tidy(() => {
    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));
    return [a, b];
  });

  console.log('before training......');
  a.print();
  b.print();

  drawLine(ctx, a.dataSync(), b.dataSync(), '#0000FF');

  // 训练迭代的次数
  const learningRate = 0.5;
  const optimizer = tf.train.adam(learningRate);

  // 开始训练
  for (let i = 0; i < numIterations; i++) {
    optimizer.minimize(() => {
      const pred = predict(trainingData.xs, { a, b });
      const error = pred.sub(trainingData.ys).square().mean();
      return error;
    });

    await tf.nextFrame();
  }

  drawLine(ctx, a.dataSync(), b.dataSync(), '#FF0000');
  console.log('after training......');
  a.print();
  b.print();

  trainingData.xs.dispose();
  trainingData.ys.dispose();
  optimizer.dispose();
  a.dispose();
  b.dispose();

  console.log(tf.memory().numTensors);

}
