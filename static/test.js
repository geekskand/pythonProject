const video = document.getElementById('webcam-video');
const submitButton = document.getElementById('submit-button');
const placeholderImage = document.getElementById('placeholder-image');
const webcamButton = document.getElementById('webcam-button');
const clearButton = document.getElementById('clear-button');

webcamButton.addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
            video.srcObject = stream;
            video.play();

            video.addEventListener('loadedmetadata', () => {
                placeholderImage.style.display = 'none';
            });

            submitButton.addEventListener('click', () => {
    // Capture image from webcam
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    // Send the captured image to the Flask server
    const capturedImage = canvas.toDataURL('image/jpeg').split(',')[1];
    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({ image: capturedImage }),
        headers: { 'Content-Type': 'application/json' }
    })
   .then(response => response.json())
   .then(data => {
        console.log(`Prediction: ${data.prediction}`);
        // Update the UI with the prediction result
         document.getElementById('chat-input-2').value = data.prediction;
    document.getElementById('output-letter').textContent = data.prediction;
    })
   .catch(error => {
        console.error('Error sending image to server:', error);
    });
});
        })
      .catch(error => {
            console.error('Error accessing webcam:', error);
        });
});
// Add event listener to clear button
clearButton.addEventListener('click', () => {
    document.getElementById('chat-input-2').value = ''; // clear input field
    document.getElementById('output-letter').textContent = ''; // clear output text
});