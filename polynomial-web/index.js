const tf = require('@tensorflow/tfjs');
const echarts = require('echarts');

const numIterations = 75;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

// 生成随机数据
function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a),
      tf.scalar(coeff.b),
      tf.scalar(coeff.c),
      tf.scalar(coeff.d)
    ];

    const xs = tf.randomUniform([numPoints], -1, 1);

    const ys = a.mul(xs.pow(tf.scalar(3)))
      .add(b.mul(xs.square()))
      .add(c.mul(xs)).add(d).add(tf.randomUniform([numPoints], 0, sigma));

    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);

    return {
      xs,
      ys: ysNormalized
    };
  });
}

// 渲染点数据
async function renderDotData(mycharts, data, title) {
  let xsData = await data.xs.data();
  let ysData = await data.ys.data();
  const dotData = Array.from(xsData).map((item, i) => {
    return [item, ysData[i]];
  });
  const option = {
    title: {
      text: title
    },
    xAxis: {},
    yAxis: {},
    series: [{
      symbolSize: 5,
      data: dotData,
      type: 'scatter'
    }]
  };
  mycharts.setOption(option);
}

// 渲染点数据和曲线数据
async function renderDotAndLine(mycharts, data, y, title) {
  let xsData = await data.xs.data();
  let ysData = await data.ys.data();
  let yPredictData = await y.data();
  const dotData = [];
  let lineData = [];
  let xsArray = Array.from(xsData);
  for (let i = 0; i < xsArray.length; i++) {
    let item = xsArray[i];
    dotData.push([item, ysData[i]]);
    lineData.push([item, yPredictData[i]]);
  }
  lineData = lineData.sort((a, b) => {
    return a[0] - b[0];
  });
  const option = {
    title: {
      text: title
    },
    xAxis: {
      type: 'value'
    },
    yAxis: {
      type: 'value'
    },
    series: [{
      symbolSize: 5,
      data: dotData,
      type: 'scatter'
    }, {
      data: lineData,
      type: 'line',
      smooth: true
    }]
  };
  mycharts.setOption(option);
}

// 函数计算
function predict({
  a, b, c, d
}, x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3, 'int32')))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

// 损失函数
function loss(prediction, labels) {
  // 预测值和实际值差值的平方的平均值
  const error = prediction.sub(labels).square().mean();
  return error;
}

// 训练
async function train(data, { a, b, c, d }) {
  for (let i = 0; i < numIterations; i++) {
    optimizer.minimize(() => {
      const pred = predict({ a, b, c, d }, data.xs);
      return loss(pred, data.ys);
    });
    await tf.nextFrame();
  }
}

window.onload = async function () {

  // 需要拟合的原始的多项式的参数
  const trueCoefficients = {
    a: -0.8,
    b: -0.2,
    c: 0.9,
    d: 0.5
  };

  const trainData = generateData(100, trueCoefficients);
  const trainCharts = echarts.init(document.getElementById('trainContainer'));
  await renderDotData(trainCharts, trainData, '拟合点数据');

  const a = tf.variable(tf.scalar(Math.random()));
  const b = tf.variable(tf.scalar(Math.random()));
  const c = tf.variable(tf.scalar(Math.random()));
  const d = tf.variable(tf.scalar(Math.random()));

  const yBeforeTrain = predict({
    a, b, c, d
  }, trainData.xs);

  const randomCharts = echarts.init(document.getElementById('randomContainer'));
  await renderDotAndLine(randomCharts, trainData, yBeforeTrain, '随机曲线');

  await train(trainData, { a, b, c, d });

  const yTrained = predict({ a, b, c, d }, trainData.xs);
  const resultCharts = echarts.init(document.getElementById('resultContainer'));
  await renderDotAndLine(resultCharts, trainData, yTrained, '训练后');

  trainData.xs.dispose();
  trainData.ys.dispose();
  yBeforeTrain.dispose();
  yTrained.dispose();

};
