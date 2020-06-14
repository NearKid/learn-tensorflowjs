const tf = require('@tensorflow/tfjs');


const a = tf.scalar(3.1415);
console.log('a:');
console.log(a.shape);
console.log(a.size);

const b = tf.tensor([3.1415]);
console.log('b:');
console.log(b.shape);
console.log(b.size);

const c = tf.tensor([3.1415, 1.23456]);
console.log('c:');
console.log(c.shape);
console.log(c.size);

const d = tf.tensor([[2, 3, 4], [5, 6, 7]]);
console.log('d:');
console.log(d.shape);
console.log(d.size);

const e = tf.tensor2d([[2, 3, 4], [5, 6, 7]]);
console.log('e:');
console.log(e.shape);
console.log(e.size);

const f = tf.tensor([2, 3, 4, 5, 6, 7], [2, 3]);
console.log('f:');
console.log(f.shape);
console.log(f.size);