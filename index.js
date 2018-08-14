import * as tf from '@tensorflow/tfjs';

// import {MnistData} from './data';

// const model = await tf.loadModel('http://localhost:8080/model/model.json');

let model = tf.sequential();

async function loadModel() {
    console.log('Loading model');
    const model = await tf.loadModel('http://localhost:8080/model/model.json');
    console.log('Model loaded');
  console.log(model);
  return model;
};

loadModel()
  .then( (modelo) => {
    // do other stuff

    console.log(modelo);  //this line produces the error below
    model = modelo;
  });

// model.add(tf.layers.conv2d({
//   inputShape: [28, 28, 1],
//   kernelSize: 5,
//   filters: 8,
//   strides: 1,
//   activation: 'relu',
//   kernelInitializer: 'varianceScaling'
// }));

// model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

// model.add(tf.layers.conv2d({
//   kernelSize: 5,
//   filters: 16,
//   strides: 1,
//   activation: 'relu',
//   kernelInitializer: 'varianceScaling'
// }));

// model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

// model.add(tf.layers.flatten());

// model.add(tf.layers.dense(
//     {units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));


// const LEARNING_RATE = 0.15;

// const optimizer = tf.train.sgd(LEARNING_RATE);


// model.compile({
//   optimizer: optimizer,
//   loss: 'categoricalCrossentropy',
//   metrics: ['accuracy'],
// });


// const BATCH_SIZE = 64;

// const TRAIN_BATCHES = 150;

// const TEST_BATCH_SIZE = 1000;

// const TEST_ITERATION_FREQUENCY = 5;

// async function train() {

//   const lossValues = [];
//   const accuracyValues = [];

//   for (let i = 0; i < TRAIN_BATCHES; i++) {
//     const [batch, validationData] = tf.tidy(() => {
//       const batch = data.nextTrainBatch(BATCH_SIZE);
//       batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);

//       let validationData;

//       if (i % TEST_ITERATION_FREQUENCY === 0) {
//         const testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
//         validationData = [
          
//           testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
//         ];
//       }
//       return [batch, validationData];
//     });

//     const history = await model.fit(
//         batch.xs, batch.labels,
//         {batchSize: BATCH_SIZE, validationData, epochs: 1});

//     const loss = history.history.loss[0];
//     const accuracy = history.history.acc[0];

//     console.log(loss, accuracy);
//   }
//   console.log("Trained!!");
// }
const emnist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
async function showPredictions(imageData) {

  const pred = await tf.tidy(() => {
    
    let img = tf.fromPixels(imageData, 1);
    img = img.reshape([1, 28, 28, 1]);
    // img = img.reshape([-1, 28 * 28]);
    img = tf.cast(img, 'float32');
    console.log("img: " + img);

    const output = model.predict(img);

    let predictions = Array.from(output.dataSync());
    console.log(predictions);
    console.log(indexOfMax(predictions));
    console.log(emnist[indexOfMax(predictions)]);
  });
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
  document.getElementById("status").innerHTML = "Data loaded";
}

async function mnist() {
  await load();
  await train();

  // showPredictions();
}
// mnist(); ####################################

$('#ejecutar').click(function(){
  console.log("ejecutando");


  // ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx2.drawImage(canvas, -14, -14, 28, 28);
  // ctx2.rotate((-1) * Math.PI/2);
  let imageData = ctx2.getImageData(0, 0, 28, 28);
  showPredictions(imageData);
});
$('#limpiar').click(function(){console.log("limpiando");clearArea();
});





let canvas;
canvas = document.getElementById('myCanvas');
let canvas2 = document.getElementById('myCanvas2');
if(typeof G_vmlCanvasManager != 'undefined') {
  canvas = G_vmlCanvasManager.initElement(canvas);
}
let mousePressed = false;
let lastX, lastY;
let ctx;
let ctx2;

function InitThis() {
    ctx = canvas.getContext("2d");
    ctx2 = canvas2.getContext("2d");
    ctx2.translate(14, 14);
    ctx2.rotate(3*Math.PI/2);
    ctx2.scale(-1, 1);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx2.fillStyle = "black";
    ctx2.fillRect(0, 0, canvas.width, canvas.height);

    $('#myCanvas').mousedown(function (e) {
        mousePressed = true;
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas').mousemove(function (e) {
        if (mousePressed) {
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas').mouseup(function (e) {
        mousePressed = false;
    });
      $('#myCanvas').mouseleave(function (e) {
        mousePressed = false;
    });
}

function Draw(x, y, isDown) {
    if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = "#f6f6f6";
        ctx.lineWidth = 35;
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x; lastY = y;
}
  
function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx2.setTransform(1, 0, 0, 1, 0, 0);
    ctx2.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx2.translate(14, 14);
    ctx2.rotate(3*Math.PI/2);
    ctx2.scale(-1, 1);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx2.fillStyle = "black";
    ctx2.fillRect(0, 0, canvas.width, canvas.height);
}

InitThis(); //################################################



function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}