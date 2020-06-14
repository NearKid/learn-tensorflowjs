const tf = require('@tensorflow/tfjs');
function main() {
  tf.tidy(() => {
    // 定义两个2x3的矩阵
    const a = tf.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor([1, 4, 7, 2, 5, 8], [2, 3]);

    // 矩阵的加法就是对应位置的元素相加
    console.log('加法......')
    // 相加方法1
    const addResult0 = tf.add(a, b);
    addResult0.print();
    // 相加方法2
    const addResult1 = a.add(b);
    addResult1.print()

    // 矩阵的减法就是对应位置的元素相减
    console.log('减法......')
    // 相减方法1
    const subResult0 = tf.sub(a, b);
    subResult0.print();
    // 相减方法2
    const subResult1 = a.sub(b);
    subResult1.print();

    // 乘法，这里的乘法是对应位置元素相乘，和矩阵的点乘有点区别，后面有讲解点乘
    console.log('减法......');
    // 相乘方法1
    const mulResult0 = tf.mul(a, b);
    mulResult0.print();
    // 相乘方法2
    const mulResult1 = a.mul(b);
    mulResult1.print();
    // 特别说明，加法和乘法都需要两个tensor的shape一直才能进行运算
    // 但是乘法可以和数字进行运算
    console.log('tensor和js数字进行相乘')
    a.mul(3).print(); 

    // 除法，也是对应位置相除
    console.log('除法......');
    // 相除方法1
    const divResult0 = tf.div(a, b);
    divResult0.print();
    // 相除方法2
    const divResult1 = a.div(b);
    divResult1.print();


  })
}

function dotMul() {
  console.log('点乘......')
  const a = tf.tensor1d([1, 2]);
  const b = tf.tensor2d([[1, 2], [3, 4]]);
  const c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);

  console.log(a.shape); // a的shape是[2]，既可以当做[2, 1], 也可以当做[1, 2]
  console.log(b.shape);
  console.log(c.shape);
  a.dot(b).print();  // 此处a作为1x2的矩阵使用，和2x2的b进行点乘即可得到结果
  b.dot(a).print(); // 此处a作为2x1的矩阵，2x2的b点乘2x1的a进行运算即可得到结果
  b.dot(c).print(); // 正常的矩阵点乘
}

// 加减乘除的demo
main();

// 矩阵点乘的demo
dotMul();
