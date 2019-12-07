const tf = require('@tensorflow/tfjs');
const MODEL_URL = 'http://localhost:8000/mnist-web-model-load/mnist-model.json';

async function getModel() {
  const model = await tf.loadLayersModel(MODEL_URL);
  return model;
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

  let imgList = document.getElementById('imgList');
  let fileElem = document.getElementById('file');
  console.log('加载模型中...');
  const model = await getModel();
  console.log('加载模型成功...');
  model.summary();

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

