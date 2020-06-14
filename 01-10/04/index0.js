// 不释放内存
const tf = require('@tensorflow/tfjs');

for(let i = 0; i < 5; i++) {
  const a = tf.tensor([i]);
  a.print();
}

// 输出5
console.log(`tensor变量的个数${tf.memory().numTensors}`)
