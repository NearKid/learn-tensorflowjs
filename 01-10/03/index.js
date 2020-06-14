const tf = require('@tensorflow/tfjs');

const a = tf.tensor([1, 2, 3]);
// 能看到一些基本信息，比如shape，但是看不到数据
console.log(a);

// 同步获取数据，数据量小的时候可以使用，数据量大的话会把主进程卡住，不推荐使用
const a_dataSync = a.dataSync();
console.log('同步获取数据：')
console.log(a_dataSync)

// 异步获取数据，推荐使用
a.data().then(data => {
  console.log('异步获取数据：')
  console.log(data);
});