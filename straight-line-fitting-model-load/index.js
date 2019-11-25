const tf = require('@tensorflow/tfjs');
const MODEL_URL = 'http://localhost:8000/straight-line-fitting-model-load/straight-line-model.json';

async function drawPoints(ctx, xs, ys, color = '#000000', r = 1) {
  ctx.fillStyle = color;
  let xValues = await xs.data();
  let yValues = await ys.data();
  ctx.beginPath();
  for (let i = 0, len = xValues.length; i < len; i++) {
    ctx.arc(xValues[i], yValues[i], r, 0, Math.PI * 2);
  }
  ctx.closePath();
  ctx.fill();
}

async function loadModel() {
  const model = await tf.loadLayersModel(MODEL_URL);
  model.summary();
  return model;
}

window.onload = async function () {
  const xDom = document.getElementById('x');
  const yDom = document.getElementById('y');
  const btnDom = document.getElementById('btn');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const model = await loadModel();

  btnDom.addEventListener('click', async function (e) {
    let x = tf.tensor([+xDom.value], [1, 1]);
    let y = model.predict(x);
    y.print();
    yDom.value = y.dataSync();
    await drawPoints(ctx, x, y, '#FF0000', 3);
    x.dispose();
    y.dispose();
  });
}
