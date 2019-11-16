const tf = require('@tensorflow/tfjs');
import {
  generateData,
  drawPoints
} from './utils';

export default async function run(inputA, inputB, randomNumberCount, numIterations, sigma, ctx) {

  const trueCoefficients = {
    a: inputA, b: inputB
  };
  const trainingData = generateData(randomNumberCount, trueCoefficients, true, sigma);

  // 画训练数据点
  await drawPoints(ctx, trainingData.xs, trainingData.ys);

  // 定义模型
  const model = tf.sequential();

  // 添加层
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
  }));

  // 传入优化器并编译模型
  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'adam'
  });

  // 训练
  await model.fit(trainingData.xs, trainingData.ys, {
    epochs: numIterations
  });

  // 测试数据
  let testX = tf.tensor([100, 200, 300, 400, 500, 600, 700], [7, 1]);

  // 预测结果
  let testY = model.predict(testX);

  // 画预测结果点
  await drawPoints(ctx, testX, testY, '#FF0000', 3);

  trainingData.xs.dispose();
  trainingData.ys.dispose();
  testX.dispose();
  testY.dispose();

}

