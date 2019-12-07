const tf = require('@tensorflow/tfjs');
const { IMAGE_H, IMAGE_W, MnistData } = require('./data');

function getModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Repeat another conv2d + maxPooling stack. 
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));


  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

function train(model, trainData) {
  const BATCH_SIZE = 512;
  const validationSplit = 0.15;
  const TRAIN_EPOCHS = 10;

  const batchDom = document.getElementById('batch');
  const epochDom = document.getElementById('epoch');

  let batchCount = 0;
  let batchTotalCount = Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / BATCH_SIZE) * TRAIN_EPOCHS;


  return model.fit(trainData.xs, trainData.labels, {
    batchSize: BATCH_SIZE,
    epochs: TRAIN_EPOCHS,
    shuffle: true,
    validationSplit,
    callbacks: {
      async onBatchEnd(batch1, { acc, batch, loss, size }) {
        batchCount++;
        batchDom.innerText = `${(batchCount / batchTotalCount * 100).toFixed(1)}%`;
        await tf.nextFrame();
      },
      async onEpochEnd(epoch, { acc, loss }) {
        epochDom.innerText = `${(epoch + 1) * 100 / TRAIN_EPOCHS}%`;
        await tf.nextFrame();
      }
    }
  });
}

function getGrayScaleData(img) {
  let canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  let ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  let imgData = ctx.getImageData(0, 0, img.width, img.height);
  return imgData;
}

function getImageByFile(file) {
  return new Promise((resolve, reject) => {
    let reader = new FileReader();
    reader.onload = function (e) {
      let img = new Image();
      img.onload = function () {
        resolve(this);
      }
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

window.onload = async function () {
  const data = new MnistData();

  let fileElem = document.getElementById('file');
  let imgList = document.getElementById('imgList');

  const btnLoad = document.getElementById('btnLoad');
  const btnTrain = document.getElementById('btnTrain');
  const loadStatus = document.getElementById('loadStatus');
  const trainStatus = document.getElementById('trainStatus');
  const btnShow = document.getElementById('btnShow');
  const saveModel = document.getElementById('saveModel');


  const model = getModel();
  model.summary();

  let trainData;

  let hasLoadData = false;
  let hasTrained = false;

  // 加载数据
  btnLoad.onclick = async function () {
    loadStatus.innerText = '加载中';
    hasLoadData = false;
    await data.load();
    hasLoadData = true;
    loadStatus.innerText = '加载完成';
    trainData = data.getTrainData();
    btnShow.innerText = '显示训练数据：' + trainData.xs.shape[0] + '中的10个';
  }

  // 显示图片数据
  btnShow.onclick = async function () {
    if (hasLoadData) {
      let startIndex = parseInt(document.getElementById('startIndex').value);
      let d = await trainData.xs.slice([startIndex], [10]).array();
      let y = await trainData.labels.slice([startIndex], [10]).array();
      let f = document.createDocumentFragment();
      for (let k = 0; k < 10; k++) {
        let canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style.margin = '4px';
        let ctx = canvas.getContext('2d');
        let imgData = ctx.createImageData(28, 28);
        let count = 0;
        for (let i = 0; i < 28; i++) {
          for (let j = 0; j < 28; j++) {
            let tmp = d[k][i][j][0] * 255;
            imgData.data[count] = tmp;
            imgData.data[count + 1] = tmp;
            imgData.data[count + 2] = tmp;
            imgData.data[count + 3] = 255;
            count += 4;
          }
        }
        ctx.putImageData(imgData, 0, 0);
        let index = 0;
        for (let i = 0; i < 10; i++) {
          if (y[k][i] === 1) {
            index = i;
            break;
          }
        }
        let div = document.createElement('div');
        div.className = 'item';
        let span = document.createElement('span');
        span.innerText = index;
        div.appendChild(canvas);
        div.appendChild(span);
        f.appendChild(div);
      }
      document.getElementById('canvasContainer').appendChild(f);
    }
  }

  // 训练数据
  btnTrain.onclick = async function () {
    if (!hasLoadData) {
      return;
    }
    trainStatus.innerText = '训练中';
    hasTrained = false;

    await train(model, trainData);
    // 使用原有的测试数据也进行训练
    // const testData = data.getTestData();
    // await train(model, testData);

    hasTrained = true;
    trainStatus.innerText = '训练完成';

  }

  // 保存模型
  saveModel.onclick = async function () {
    await model.save('downloads://mnist-model');
    console.log('保存成功');
  }

  // 选择文件
  fileElem.onchange = async function () {
    let file = this.files[0];

    let img = await getImageByFile(file);
    let d = getGrayScaleData(img);
    let predictResult = tf.tidy(() => {
      let t = tf.tensor(new Uint8Array(d.data), [d.height, d.width, 4]);
      let img = t.split([3, 1], -1)[0];
      let nimg = tf.image.resizeBilinear(img.expandDims(0), [28, 28]);
      let [a, b, c] = nimg.split([1, 1, 1], 3);
      let gray = a.mul(0.299).add(b.mul(0.587)).add(c.mul(0.114));
      let res = model.predict(gray);
      return res;
    });

    let data = await predictResult.data();
    let result = 0;
    console.log(data);
    for (let i = 0; i < 10; i++) {
      if (data[i] === 1) {
        result = i;
        break;
      }
    }
    this.value = '';
    let item = document.createElement('div');
    item.className = 'img-item';
    item.appendChild(img);
    let span = document.createElement('span');
    span.innerText = result;
    item.appendChild(span);
    imgList.appendChild(item);
  }

}

