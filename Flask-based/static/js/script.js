// --- static/js/script.js ---

// Grab DOM elements
const canvas     = document.getElementById('canvas');
const ctx        = canvas.getContext('2d');
const clearBtn   = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predEl     = document.getElementById('prediction');
const confEl     = document.getElementById('confidence');
const top3El     = document.getElementById('top3');
const chartCtx   = document.getElementById('chart').getContext('2d');

// Initialize white background
ctx.fillStyle = '#fff';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing style
ctx.lineCap   = 'round';
ctx.lineWidth = 20;
ctx.strokeStyle = '#000';

// Track whether we're currently drawing
let drawing = false;

// Utility: get x/y from mouse or touch event
function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  if (e.touches) {
    return {
      x: e.touches[0].clientX - rect.left,
      y: e.touches[0].clientY - rect.top
    };
  } else {
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }
}

// Start drawing
function startDraw(e) {
  e.preventDefault();
  drawing = true;
  draw(e);  // draw a dot immediately
}

// End drawing
function endDraw(e) {
  e.preventDefault();
  drawing = false;
  ctx.beginPath();  // reset path so lines don't connect
}

// Draw line
function draw(e) {
  if (!drawing) return;
  e.preventDefault();
  const pos = getPos(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

// Mouse events
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup',   endDraw);
canvas.addEventListener('mouseout',  endDraw);

// Touch events
canvas.addEventListener('touchstart', startDraw, { passive: false });
canvas.addEventListener('touchmove',  draw,      { passive: false });
canvas.addEventListener('touchend',   endDraw,   { passive: false });

// Initialize Chart.js with a single instance
const chart = new Chart(chartCtx, {
  type: 'bar',
  data: {
    labels: Array.from({length: 10}, (_, i) => i),
    datasets: [{
      label: 'Probability',
      data: Array(10).fill(0),
      backgroundColor: Array(10).fill('#90CAF9')
    }]
  },
  options: {
    animation: false,
    scales: {
      y: { beginAtZero: true, max: 1 }
    },
    plugins: {
      legend: { display: false }
    }
  }
});

// Reset textual and chart results
function resetResults() {
  // Clear the processed image
  const preprocessedImg = document.getElementById('preprocessed');
  if (preprocessedImg.hasAttribute('src')) {
    preprocessedImg.removeAttribute('src');
  }
  
  // Clear prediction results
  predEl.textContent = '';
  confEl.textContent = '';
  top3El.innerHTML = '';
  
  // Reset chart
  chart.data.datasets[0].data = Array(10).fill(0);
  chart.data.datasets[0].backgroundColor = Array(10).fill('#90CAF9');
  chart.update();
  chartCtx.canvas.classList.remove('has-data');
}

// Display server response
function showResults(data) {
  // Display processed image
  const preprocessedImg = document.getElementById('preprocessed');
  preprocessedImg.src = data.processedImage;

  // Text
  predEl.textContent = data.prediction;
  confEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  // Top-3 list
  top3El.innerHTML = '';
  data.top3.forEach((t, i) => {
    const li = document.createElement('li');
    li.textContent = `${i+1}. Digit ${t.digit} — ${(t.confidence * 100).toFixed(1)}%`;
    top3El.appendChild(li);
  });

  // Chart update
  chart.data.datasets[0].data = data.probabilities;
  chart.data.datasets[0].backgroundColor = data.probabilities.map((p, i) =>
    i === data.prediction ? '#1976D2' : '#90CAF9'
  );
  chart.update();
  chartCtx.canvas.classList.add('has-data');
}

// Check if canvas is blank (all-white)
function isCanvasBlank() {
  const pixelData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  // If every R,G,B,A pixel is (255,255,255,255) or (255,255,255,0) depending,
  // sum will be max value. We'll check a subset for speed.
  for (let i = 0; i < pixelData.length; i += 4 * 10) {
    if (pixelData[i] < 250) return false;
  }
  return true;
}

// Clear button
clearBtn.onclick = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resetResults();
};

// Predict button
predictBtn.onclick = async () => {
  if (isCanvasBlank()) {
    alert('Please draw a digit before predicting!');
    return;
  }

  const dataUrl = canvas.toDataURL('image/png');
  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ imageData: dataUrl })
  });
  const data = await res.json();
  showResults(data);
};
