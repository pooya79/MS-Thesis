const form = document.querySelector('#transcription-form');
const fileInput = document.querySelector('#audio-file');
const dropZone = document.querySelector('#drop-zone');
const preview = document.querySelector('#audio-preview');
const player = document.querySelector('#audio-player');
const submit = document.querySelector('#submit-button');
const errorBox = document.querySelector('#form-error');
const loadingPanel = document.querySelector('#loading-panel');
const loadingTitle = document.querySelector('#loading-title');
const loadingDetail = document.querySelector('#loading-detail');
const maxAudioSeconds = Number(form.dataset.maxAudioSeconds);
let selectedAudio = null;
let objectUrl = null;
let mediaRecorder = null;
let timer = null;
let elapsed = 0;
let animationFrame = null;

document.querySelectorAll('.source-tab').forEach((tab) => tab.addEventListener('click', () => {
  document.querySelectorAll('.source-tab').forEach((item) => { item.classList.toggle('active', item === tab); item.setAttribute('aria-selected', item === tab); });
  document.querySelector('#upload-panel').classList.toggle('hidden', tab.dataset.tab !== 'upload');
  document.querySelector('#record-panel').classList.toggle('hidden', tab.dataset.tab !== 'record');
}));

function useAudio(file, label = file.name) {
  selectedAudio = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  player.src = objectUrl;
  document.querySelector('#audio-name').textContent = label;
  document.querySelector('#audio-meta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  preview.classList.remove('hidden');
  submit.disabled = false;
  errorBox.textContent = '';
}

fileInput.addEventListener('change', () => { if (fileInput.files[0]) useAudio(fileInput.files[0]); });
['dragenter', 'dragover'].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => { const file = event.dataTransfer.files[0]; if (file) useAudio(file); });
document.querySelector('#remove-audio').addEventListener('click', () => { selectedAudio = null; fileInput.value = ''; player.removeAttribute('src'); preview.classList.add('hidden'); submit.disabled = true; });

const recordButton = document.querySelector('#record-button');
recordButton.addEventListener('click', async () => {
  if (mediaRecorder?.state === 'recording') { mediaRecorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    audioContext.createMediaStreamSource(stream).connect(analyser);
    const canvas = document.querySelector('#visualizer');
    const context = canvas.getContext('2d');
    const levels = new Uint8Array(analyser.frequencyBinCount);
    const drawLevels = () => {
      analyser.getByteFrequencyData(levels); context.clearRect(0, 0, canvas.width, canvas.height);
      const width = canvas.width / levels.length;
      levels.forEach((level, index) => {
        const height = Math.max(2, level / 255 * canvas.height);
        context.fillStyle = '#516052'; context.fillRect(index * width, (canvas.height - height) / 2, Math.max(1, width - 2), height);
      });
      animationFrame = requestAnimationFrame(drawLevels);
    };
    drawLevels();
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.addEventListener('dataavailable', (e) => chunks.push(e.data));
    mediaRecorder.addEventListener('stop', () => {
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      useAudio(new File([blob], 'recording.webm', { type: blob.type }), 'Browser recording');
      stream.getTracks().forEach((track) => track.stop()); clearInterval(timer); cancelAnimationFrame(animationFrame); audioContext.close();
      recordButton.classList.remove('recording'); recordButton.innerHTML = '<span></span> Start recording';
      document.querySelector('#record-dot').classList.remove('live'); document.querySelector('#record-label').textContent = 'Recording ready';
    });
    elapsed = 0; mediaRecorder.start(); document.querySelector('#record-dot').classList.add('live');
    document.querySelector('#record-label').textContent = 'Recording'; recordButton.classList.add('recording'); recordButton.innerHTML = '<span></span> Stop recording';
    timer = setInterval(() => {
      elapsed += 1;
      const minutes = Math.floor(elapsed / 60); const seconds = elapsed % 60;
      document.querySelector('#record-time').textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
      if (elapsed >= maxAudioSeconds) mediaRecorder.stop();
    }, 1000);
  } catch (_) { errorBox.textContent = 'Microphone access was not granted. Check your browser permissions.'; }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault(); if (!selectedAudio) return;
  submit.disabled = true; submit.classList.add('loading'); errorBox.textContent = '';
  loadingPanel.classList.remove('hidden');
  const modelId = new FormData(form).get('model_id');
  const body = new FormData(); body.append('model_id', modelId); body.append('audio', selectedAudio);
  const updateProgress = async () => {
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/status`);
      if (!response.ok) return;
      const status = await response.json();
      const device = status.device === 'cuda' ? 'GPU' : 'CPU';
      if (status.state === 'ready') {
        loadingTitle.textContent = `Transcribing on ${device}`;
        loadingDetail.textContent = `${status.label} is ready and processing your audio.`;
      } else {
        loadingTitle.textContent = `Loading ${status.label} on ${device}`;
        loadingDetail.textContent = 'The first run can take longer while the checkpoint is loaded.';
      }
    } catch (_) { loadingTitle.textContent = 'Preparing model'; }
  };
  await updateProgress();
  const statusTimer = setInterval(updateProgress, 400);
  try {
    const response = await fetch('/api/transcriptions', { method: 'POST', body });
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Transcription failed');
    document.querySelector('#result-text').textContent = data.text || 'No speech was detected.';
    const device = data.device === 'cuda' ? 'GPU' : 'CPU';
    document.querySelector('#result-meta').innerHTML = `<span>${data.model_id}</span><span>${device}</span><span>${data.duration_seconds}s audio</span><span>${data.processing_seconds}s processing</span>`;
    const result = document.querySelector('#result'); result.classList.remove('hidden'); result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) { errorBox.textContent = error.message; }
  finally { clearInterval(statusTimer); loadingPanel.classList.add('hidden'); submit.disabled = false; submit.classList.remove('loading'); }
});

document.querySelector('#copy-button').addEventListener('click', async (event) => {
  await navigator.clipboard.writeText(document.querySelector('#result-text').textContent);
  event.currentTarget.textContent = 'Copied'; setTimeout(() => { event.currentTarget.textContent = 'Copy text'; }, 1400);
});
