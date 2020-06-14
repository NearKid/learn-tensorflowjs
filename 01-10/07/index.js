const tf = require('@tensorflow/tfjs');

const a = tf.scalar(1);
const b = tf.scalar(2);
const c = tf.scalar(3);

// 计算 y = a*(x^2) + b*x + c
function predict(x) {
  return tf.tidy(() => {
    if (typeof x == 'number') {
      x = tf.scalar(x);
    }
    return a.mul(x.square())
    .add(b.mul(x))
    .add(c);
  });
}

const result = predict(3);
result.print()