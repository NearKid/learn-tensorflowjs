import run from './run.js';

window.onload = function () {
  let aDom = document.getElementById('a');
  let bDom = document.getElementById('b');
  let randomNumberCountDom = document.getElementById('randomNumberCount');
  let numIterationsDom = document.getElementById('numIterations');
  const sigmaDom = document.getElementById('sigma');
  let btn = document.getElementById('btn');
  let canClick = true;

  let ctx = document.getElementById('canvas').getContext('2d');

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
    await run(a, b, randomNumberCount, numIterations, sigma, ctx);
    canClick = true;
  }

};
