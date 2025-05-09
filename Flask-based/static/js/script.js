const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// initialize white background
ctx.fillStyle = '#fff';
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;
ctx.lineCap = 'round';
ctx.lineWidth = 20;
ctx.strokeStyle = '#000';

canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup',   () => drawing = false);
canvas.addEventListener('mouseout',  () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
}

document.getElementById('clearBtn').onclick = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resetResults();
};

document.getElementById('predictBtn').onclick = async () => {
  const dataUrl = canvas.toDataURL('image/png');
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageData: dataUrl })
  });
  const data = await res.json();
  showResults(data);
};

function resetResults() {
  document.getElementById('prediction').textContent = '';
  document.getElementById('confidence').textContent = '';
  document.getElementById('top3').innerHTML = '';
  const chartEl = document.getElementById('chart');
  chartEl.getContext('2d').clearRect(0,0,chartEl.width, chartEl.height);
}

function showResults(data) {
  document.getElementById('prediction').textContent = data.prediction;
  document.getElementById('confidence').textContent = 'Confidence: ' + data.confidence.toFixed(2);

  // Top3
  const top3 = document.getElementById('top3');
  top3.innerHTML = '';
  data.top3.forEach((t, i) => {
    const li = document.createElement('li');
    li.textContent = `${i+1}. Digit ${t.digit} — ${t.confidence.toFixed(2)}`;
    top3.appendChild(li);
  });

  // Chart.js bar chart
  const ctxChart = document.getElementById('chart').getContext('2d');
  new Chart(ctxChart, {
    type: 'bar',
    data: {
      labels: Array.from(Array(10).keys()),
      datasets: [{
        label: 'Probability',
        data: data.probabilities,
        backgroundColor: data.probabilities.map((p,i) => i===data.prediction ? '#1976D2' : '#90CAF9')
      }]
    },
    options: {
      scales: { y: { beginAtZero: true, max: 1 } }
    }
  });
}
