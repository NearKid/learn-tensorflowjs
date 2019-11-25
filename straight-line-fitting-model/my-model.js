const tf = require('@tensorflow/tfjs');
import {
  generateData,
  drawPoints
} from './utils';

export default class MyModel {
  constructor(ctx) {
    this.ctx = ctx;

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

    this.model = model;
  }
  async train(inputA, inputB, randomNumberCount, numIterations, sigma) {

    const trueCoefficients = {
      a: inputA, b: inputB
    };
    const trainingData = generateData(randomNumberCount, trueCoefficients, true, sigma);

    // 画训练数据点
    await drawPoints(this.ctx, trainingData.xs, trainingData.ys);

    // 训练
    await this.model.fit(trainingData.xs, trainingData.ys, {
      epochs: numIterations
    });

    trainingData.xs.dispose();
    trainingData.ys.dispose();

  }

  async save() {
    const saveResults = await this.model.save('downloads://straight-line-model');
    console.log(saveResults);
  }

  async predict() {

    // 测试数据
    let testX = tf.tensor([100, 200, 300, 400, 500, 600, 700], [7, 1]);

    // 预测结果
    let testY = this.model.predict(testX);
    console.log('predict...');
    testY.print();
    // 画预测结果点
    await drawPoints(this.ctx, testX, testY, '#FF0000', 3);

    testX.dispose();
    testY.dispose();
  }


}

