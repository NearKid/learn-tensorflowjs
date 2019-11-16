const tf = require('@tensorflow/tfjs');

// canvas的宽度
export const CANW = 800;

// 生成数据
export function generateData(numPoints, coeff, reshape = false, sigma = 3) {
  return tf.tidy(() => {
    const [a, b] = [tf.scalar(coeff.a), tf.scalar(coeff.b)];

    // 生成-1到1之间的numPoints个随机数
    const xs = tf.randomUniform([numPoints], 0, CANW);

    // a*x+ b
    const ys = a.mul(xs)
      .add(b)
      .add(tf.randomUniform([numPoints], -1 * sigma, sigma));

    let xst, yst;
    if (reshape) {
      xst = xs.reshape([numPoints, 1]);
      yst = ys.reshape([numPoints, 1]);
    } else {
      xst = xs;
      yst = ys;
    }
    return {
      xs: xst, ys: yst
    }
  });
}

// 获取 y = a * x + b 结果
export function predict(x, { a, b }) {
  return tf.tidy(() => {
    return a.mul(x).add(b);
  });
}

// 画点
export async function drawPoints(ctx, xs, ys, color = '#000000', r = 1) {
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

const rate = 1000000000000000;
// 画线
export async function drawLine(ctx, a, b, color = '#000000') {
  a = parseInt(a * rate) / rate;
  b = parseInt(b * rate) / rate;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(0, b);
  ctx.lineTo(CANW, a * CANW + b);
  ctx.closePath();
  ctx.stroke();
}