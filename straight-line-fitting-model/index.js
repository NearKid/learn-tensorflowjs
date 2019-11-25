import MyModel from './my-model.js';

window.onload = function () {
  let aDom = document.getElementById('a');
  let bDom = document.getElementById('b');
  let randomNumberCountDom = document.getElementById('randomNumberCount');
  let numIterationsDom = document.getElementById('numIterations');
  const sigmaDom = document.getElementById('sigma');
  let btn = document.getElementById('btn');
  let btnSave = document.getElementById('btnSave');

  let canClick = true;
  let ctx = document.getElementById('canvas').getContext('2d');

  const myModel = new MyModel(ctx);

  btn.onclick = async function () {
    if (!canClick) {
      return;
    }
    let a = +aDom.value;
    let b = +bDom.value;
    let randomNumberCount = +randomNumberCountDom.value;
    let numIterations = +numIterationsDom.value;
    let sigma = +sigmaDom.value;
    canClick = false;
    await myModel.train(a, b, randomNumberCount, numIterations, sigma);
    await myModel.predict();
    canClick = true;
  }

  btnSave.onclick = function () {
    myModel.save();
  }

};
