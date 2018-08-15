import * as tf from '@tensorflow/tfjs';

let model = tf.sequential();
let loading = $("#loading");

async function loadModel() {
    console.log('Loading model');
    const model = await tf.loadModel('https://repounah.firebaseapp.com/model.json');
    // const model = await tf.loadModel('http://localhost:8080/model_2/model.json');
    console.log('Model loaded');
    loading.hide();
  return model;
};

loadModel()
  .then( (modelo) => {
    model = modelo;
  });

const emnist = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t'];
async function showPredictions(imageData) {

  const pred = await tf.tidy(() => {
    
    let img = tf.fromPixels(imageData, 1); 
    // img = img.reshape([1, 28, 28, 1]);   // Reshape para model_1
    img = img.reshape([-1, 28 * 28]);       // Reshape para model_2
    img = tf.cast(img, 'float32');
    console.log("img: " + img);

    const output = model.predict(img);

    let predictions = Array.from(output.dataSync());
    console.log(predictions);
    console.log(indexOfMax(predictions));
    console.log(emnist[indexOfMax(predictions)]);
    $("#prediccion").html(emnist[indexOfMax(predictions)]);

    // predictions.forEach((elem) => {
    //   return elem * 100;
    // })
    console.log(predictions);
    actualizarGrafico(predictions);
  });
}


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
let ctxChar = document.getElementById("myChart").getContext('2d');
let myChart;
let colores = ['#FF6633', '#FFB399', '#FF33FF', '#FFFF99', '#00B3E6', 
      '#E6B333', '#3366E6', '#999966', '#99FF99', '#B34D4D',
      '#80B300', '#809900', '#E6B3B3', '#6680B3', '#66991A', 
      '#FF99E6', '#CCFF1A', '#FF1A66', '#E6331A', '#33FFCC',
      '#66994D', '#B366CC', '#4D8000', '#B33300', '#CC80CC', 
      '#66664D', '#991AFF', '#E666FF', '#4DB3FF', '#1AB399',
      '#E666B3', '#33991A', '#CC9999', '#B3B31A', '#00E680', 
      '#4D8066', '#809980', '#E6FF80', '#1AFF33', '#999933',
      '#FF3380', '#CCCC00', '#66E64D', '#4D80CC', '#9900B3', 
      '#E64D66', '#4DB380']

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

  actualizarGrafico(new Array(47));

  $('#myCanvas').mousedown(function (e) {
      mousePressed = true;
      Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false, e);
  });

  $('#myCanvas').mousemove(function (e) {
      if (mousePressed) {
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true, e);
      }
  });

  $('#myCanvas').mouseup(function (e) {
      mousePressed = false;
  });
    $('#myCanvas').mouseleave(function (e) {
      mousePressed = false;
  });

  // Set up touch events for mobile
  canvas.addEventListener("touchstart", function (e) {
    e.preventDefault();
    var touch = e.touches[0];
    var mouseEvent = new MouseEvent("mousedown", {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
  }, false);
  canvas.addEventListener("touchend", function (e) {
    e.preventDefault();
    var mouseEvent = new MouseEvent("mouseup", {});
    canvas.dispatchEvent(mouseEvent);
  }, false);
  canvas.addEventListener("touchmove", function (e) {
    e.preventDefault();
    var touch = e.touches[0];
    var mouseEvent = new MouseEvent("mousemove", {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
  }, false);

}

function Draw(x, y, isDown, e) {
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

    actualizarGrafico(new Array(47));
    $("#prediccion").html("~");
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

function actualizarGrafico(data){
  if(myChart) myChart.destroy();
  myChart = new Chart(ctxChar, {
      type: 'bar',
      data: {
          labels: emnist,
          datasets: [{
              label: "Valores de predicción de la red neuronal",
              backgroundColor: colores,
              borderColor: colores,
              data: data,
              borderWidth: 1
          }]
      },
      options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: {
            duration: 2000,
            easing: 'easeInOutQuart'
          },
          scales: {
              yAxes: [{
                  ticks: {
                      beginAtZero:true
                  }
              }]
          }
      }
  });
}