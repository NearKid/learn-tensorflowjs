const tf = require('@tensorflow/tfjs');

window.onload = function () {
    const canvasInput = document.getElementById('canvasInput');
    const canvasReverse = document.getElementById('canvasReverse');

    const ctxInput = canvasInput.getContext('2d');
    const ctxReverse = canvasReverse.getContext('2d');

    let img = document.getElementById('img');
    img.onload = function() {
      console.log('onload')
    }
    ctxInput.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvasInput.width, canvasInput.height);

    let imgData = ctxInput.getImageData(0, 0, canvasInput.width, canvasInput.height);
    let canvasData = Array.prototype.slice.call(imgData.data);

    // 翻转矩阵，即图片从中心对称变换
    const reverseData = tf.tidy(() => {
        let tfCanvasData = tf.tensor(canvasData, [imgData.height, imgData.width, 4]);
        const reverseData = tfCanvasData.reverse([1,0])
        return new Uint8ClampedArray(reverseData.dataSync());
    });

    const reverseImageData = new ImageData(reverseData, imgData.width, imgData.height);
    ctxReverse.putImageData(reverseImageData, 0, 0);
}
