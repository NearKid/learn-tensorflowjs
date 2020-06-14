// 使用dispose手动释放内存
const tf = require('@tensorflow/tfjs');

// 传入的函数不能是异步的
tf.tidy(() => {
  for(let i = 0; i < 5; i++) {
    const a = tf.tensor([i]);
    a.print();
  }
})

// 输出0
console.log(`tensor变量的个数${tf.memory().numTensors}`)
