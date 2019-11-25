const tf = require('@tensorflow/tfjs');
const MODEL_URL = 'http://localhost:8000/mnist-web-model-load/mnist-model.json';

async function getModel() {
  const model = await tf.loadLayersModel(MODEL_URL);
  return model;
}

function getGrayScaleData(file) {
  return new Promise((resolve, reject) => {
    let reader = new FileReader();
    reader.onload = function (e) {
      let img = new Image();
      img.onload = function () {
        let canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        let ctx = canvas.getContext('2d');
        ctx.drawImage(this, 0, 0);

        let imgData = ctx.getImageData(0, 0, this.width, this.height);
        resolve(imgData);
      }
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

window.onload = async function () {

  let fileElem = document.getElementById('file');

  const model = await getModel();
  model.summary();

  // 选择文件
  fileElem.onchange = async function () {
    let file = this.files[0];

    let d = await getGrayScaleData(file);
    tf.tidy(() => {
      let t = tf.tensor(new Uint8Array(d.data), [d.height, d.width, 4]);
      let img = t.split([3, 1], -1)[0];
      let nimg = tf.image.resizeBilinear(img.expandDims(0), [28, 28]);
      let [a, b, c] = nimg.split([1, 1, 1], 3);
      let gray = a.mul(0.299).add(b.mul(0.587)).add(c.mul(0.114));
      let res = model.predict(gray);
      console.log(res);
      res.print();
    });

    this.value = '';
  }

}

