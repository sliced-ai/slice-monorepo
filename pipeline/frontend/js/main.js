// main.js

// DOM elements
const userInputTextarea = document.getElementById('user-input');
const submitInputBtn = document.getElementById('submit-input-btn');
const generatedDataSection = document.getElementById('generated-data');
const loadingIndicator = document.getElementById('loading-indicator');
const editArea = document.getElementById('edit-area');
const submitEditBtn = document.getElementById('submit-edit-btn');
const trainingSection = document.getElementById('training');
const trainingProgressBar = document.getElementById('training-progress-bar');
const trainingMessage = document.getElementById('training-message');
const performanceSection = document.getElementById('performance');
const performanceCharts = document.getElementById('performance-charts');
const startChatBtn = document.getElementById('start-chat-btn');
const chatSection = document.getElementById('chat');
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const sendChatBtn = document.getElementById('send-chat-btn');
const metricsSidebar = document.getElementById('metrics-sidebar');

// Event listeners
submitInputBtn.addEventListener('click', handleSubmitInput);
submitEditBtn.addEventListener('click', handleSubmitEdit);
startChatBtn.addEventListener('click', handleStartChat);
sendChatBtn.addEventListener('click', handleSendChat);

// Functions
function handleSubmitInput() {
  const userInput = userInputTextarea.value;
  sendInputToServer(userInput);
  showSection(generatedDataSection);
  showElement(loadingIndicator);
}

async function sendInputToServer(input) {
  const requestUrl = '/generate_data';

  // Show loading indicator
  showElement(loadingIndicator);

  try {
    const response = await fetch(requestUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input: input }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const characterSeed = data.character_seed || "No character seed generated";
    displayGeneratedData(characterSeed);
  } catch (error) {
    console.error('Error:', error);
    displayGeneratedData("Error generating data. Please try again.");
  } finally {
    // Hide loading indicator
    hideElement(loadingIndicator);
  }
}


function displayGeneratedData(data) {
  const editArea = document.getElementById('edit-area');
  editArea.innerHTML = ''; // Clear previous content

  const characterSeedElement = document.createElement('div');
  characterSeedElement.classList.add('character-seed');
  characterSeedElement.innerHTML = data.replace(/\n/g, '<br>'); // Replace newline characters with <br> tags
  editArea.appendChild(characterSeedElement);

  hideElement(loadingIndicator);
}

function handleSubmitEdit() {
  const editedData = editArea.value;
  sendEditedDataToServer(editedData);
  showSection(trainingSection);
  updateTrainingProgress(0);
}

function sendEditedDataToServer(data) {
  const requestUrl = '/train_model';
  fetch(requestUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ data: data }),
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      const trainingProgress = data.training_progress || 0;
      updateTrainingProgress(trainingProgress);
    })
    .catch((error) => {
      console.error('Error:', error);
    });
}

function handleStartChat() {
  showSection(chatSection);
}

function handleSendChat() {
  const userMessage = chatInput.value.trim();
  if (userMessage) {
    sendChatToServer(userMessage);
    chatInput.value = '';
  }
}

function sendChatToServer(chatText) {
  const requestUrl = '/chat';
  updateChatDisplay('user', chatText);
  fetch(requestUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message: chatText }),
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      const serverResponse = data.response || "No response";
      updateChatDisplay('ai', serverResponse);
    })
    .catch((error) => {
      updateChatDisplay('error', `Error: ${error.message}`);
    });
}

function updateChatDisplay(sender, message) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('message', sender);
  messageElement.textContent = `${sender === 'user' ? 'You' : 'AI'}: ${message}`;
  chatWindow.appendChild(messageElement);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showSection(section) {
  // Hide all sections
  const sections = document.querySelectorAll('section');
  sections.forEach(section => section.classList.add('hidden'));
  // Show the specified section
  section.classList.remove('hidden');
}

function showElement(element) {
  element.classList.remove('hidden');
}

function hideElement(element) {
  element.classList.add('hidden');
}

function updateTrainingProgress(progress) {
  trainingProgressBar.value = progress;
  // TODO: Update training message based on progress
}

// TODO: Implement functions to display model performance and metrics