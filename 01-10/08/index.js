const tf = require('@tensorflow/tfjs');

window.onload = function () {
    const canvasInput = document.getElementById('canvasInput');
    const canvasTranspose = document.getElementById('canvasTranspose');

    const ctxInput = canvasInput.getContext('2d');
    const ctxTranspose = canvasTranspose.getContext('2d');

    let img = document.getElementById('img');
    ctxInput.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvasInput.width, canvasInput.height);

    let imgData = ctxInput.getImageData(0, 0, canvasInput.width, canvasInput.height);
    let canvasData = Array.prototype.slice.call(imgData.data);

    // 转置矩阵，即沿对角线对称变换
    const transposeData = tf.tidy(() => {
        const tfCanvasData = tf.tensor(canvasData, [imgData.height, imgData.width, 4]);
        const transposeData = tfCanvasData.transpose([1, 0, 2]);
        return new Uint8ClampedArray(transposeData.dataSync());
    });

    const transposeImageData = new ImageData(transposeData, imgData.height, imgData.width);
    ctxTranspose.putImageData(transposeImageData, 0, 0);

}
