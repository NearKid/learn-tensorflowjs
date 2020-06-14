const tf = require('@tensorflow/tfjs');

// 定义张量
const a = tf.tensor([1]);
console.log(a);
a.print();
// 定义变量，只能通过张量来初始化
const b = tf.variable(a);
console.log(b)
b.print();

// 张量不可以被改变值
try {
  const updatedValues = tf.tensor([2]);
  a.assign(updatedValues);
  console.log('更新张量成功')
  a.print();
} catch (error) {
  console.log('更新张量失败')
}

// 变量可以使用assign改变值
try {
  const updatedValues = tf.tensor([2]);
  b.assign(updatedValues);
  console.log('更新变量成功')
  b.print();
} catch (error) {
  console.log('更新变量失败')
}

// 变量不可以使用不同维度的张量来改变值
try {
  const updatedValues = tf.tensor([2, 3]);
  b.assign(updatedValues);
  console.log('更新变量成功')
  b.print();
} catch (error) {
  console.log('更新变量失败')
}