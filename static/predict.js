// static/predict.js
const form = document.getElementById('predict-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(form);
  const payload = {};
  for (const [k, v] of formData.entries()) payload[k] = v;
  payload['Pclass'] = parseInt(payload['Pclass']);
  payload['Age'] = payload['Age'] ? parseFloat(payload['Age']) : null;
  payload['Fare'] = payload['Fare'] ? parseFloat(payload['Fare']) : null;

  const resp = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const json = await resp.json();
  if (json.error) {
    resultDiv.innerText = json.error;
  } else {
    resultDiv.innerText = `Survived: ${json.survived} (confidence: ${json.confidence})`;
  }
});
