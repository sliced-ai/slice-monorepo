document.addEventListener('DOMContentLoaded', function () {
    const bodyClass = localStorage.getItem('theme') || 'light-mode';
    document.body.classList.add(bodyClass);

    document.querySelectorAll('form, input, .conversation, .visualization').forEach(function (element) {
        if (bodyClass === 'dark-mode') {
            element.classList.add('dark-mode');
            element.classList.remove('light-mode');
        } else {
            element.classList.add('light-mode');
            element.classList.remove('dark-mode');
        }
    });

    document.getElementById('projects-button').addEventListener('click', function() {
        window.location.href = '/projects';
    });

    const params = new URLSearchParams(window.location.search);
    const projectName = params.get('project');
    if (projectName) {
        fetch(`/project_data/${projectName}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    populateChatWindow(data);
                }
            })
            .catch(error => console.error('Error fetching project data:', error));
    }
});

document.getElementById('home-button').addEventListener('click', function() {
    window.location.href = '/';
});

function viewProject(projectName) {
    window.location.href = '/?project=' + encodeURIComponent(projectName);
}

document.getElementById('chat-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const inputText = document.getElementById('input_text').value;
    if (inputText.trim() === "") {
        alert('Please enter some text.');
        return;
    }

    const formData = new FormData(document.getElementById('experiment-form'));
    formData.append('input_text', inputText);

    console.log('Submitting chat form with data:', Object.fromEntries(formData.entries()));

    const conversation = document.getElementById('conversation');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.innerText = inputText;
    conversation.appendChild(userMessage);

    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'message bot';
    loadingIndicator.id = 'loading-indicator';
    loadingIndicator.innerText = 'Processing...';
    conversation.appendChild(loadingIndicator);

    document.getElementById('input_text').value = '';
    conversation.scrollTop = conversation.scrollHeight;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000000);

    fetch('/chat', {
        method: 'POST',
        body: formData,
        signal: controller.signal
    })
    .then(response => response.json())
    .then(data => {
        clearTimeout(timeoutId);
        console.log('Received chat response:', data);
        if (data.error) {
            alert(data.error);
            conversation.removeChild(loadingIndicator);
            return;
        }

        const botMessage = document.createElement('div');
        botMessage.className = 'message bot';
        botMessage.id = 'bot-message';
        botMessage.innerText = data.chosen_response;
        conversation.replaceChild(botMessage, loadingIndicator);

        console.log('Fetching UMAP data from path:', data.umap_data_path);

        fetch(data.umap_data_path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(umapData => {
                console.log('Received UMAP data:', umapData);
                if (!umapData.embeddings || !Array.isArray(umapData.embeddings) || umapData.embeddings.length === 0) {
                    throw new Error('Invalid UMAP data received.');
                }
                createUMAPVisualization(umapData.embeddings, data.responses, data.responses.map((_, index) => index));
                createGridVisualization(umapData.embeddings, data.responses);
            })
            .catch(error => {
                console.error('Error fetching or processing UMAP data:', error);
                alert('Failed to fetch or process UMAP data.');
            });

        conversation.scrollTop = conversation.scrollHeight;
    })
    .catch(error => {
        clearTimeout(timeoutId);
        console.error('Error sending chat message:', error);
        alert('Failed to send message.');
        conversation.removeChild(loadingIndicator);
    });

    const statusInterval = setInterval(() => {
        fetch('/status')
            .then(response => response.json())
            .then(statusData => {
                console.log('Received status:', statusData);
                loadingIndicator.innerText = `Processing... ${statusData.status}`;
            })
            .catch(error => {
                console.error('Error fetching status:', error);
                clearInterval(statusInterval);
            });
    }, 5000);
});

document.getElementById('toggle-view').addEventListener('click', function() {
    const visualizationContainer = document.getElementById('visualization-container');
    const formContainer = document.getElementById('form-container');

    if (visualizationContainer.style.display === 'none') {
        visualizationContainer.style.display = 'block';
        formContainer.style.display = 'none';
        this.innerText = 'Show Configuration';
    } else {
        visualizationContainer.style.display = 'none';
        formContainer.style.display = 'block';
        this.innerText = 'Show Visualizations';
    }
});

document.getElementById('dark-mode-btn').addEventListener('click', function() {
    if (document.body.classList.contains('dark-mode')) {
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
        localStorage.setItem('theme', 'light-mode');
    } else {
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
        localStorage.setItem('theme', 'dark-mode');
    }
    document.querySelectorAll('form, input, .conversation, .visualization').forEach(function(element) {
        if (document.body.classList.contains('dark-mode')) {
            element.classList.add('dark-mode');
            element.classList.remove('light-mode');
        } else {
            element.classList.add('light-mode');
            element.classList.remove('dark-mode');
        }
    });
});

document.getElementById('experiment_name').addEventListener('input', function() {
    const experimentName = this.value.trim() || 'none';
    document.getElementById('displayed-experiment-name').innerText = experimentName;
});

function updateChatWindow(responseText, stepIndex) {
    console.log('Updating chat window for stepIndex:', stepIndex, 'with response:', responseText);
    const conversation = document.getElementById('conversation');
    const targetBotMessage = conversation.querySelector(`.message.bot[data-step-index="${stepIndex}"]`);
    console.log('Found targetBotMessage for stepIndex:', stepIndex, targetBotMessage);

    if (targetBotMessage) {
        console.log('Updating existing bot message for stepIndex:', stepIndex);
        targetBotMessage.innerText = responseText || 'Response not available';
    } else {
        console.log('Creating new bot message for stepIndex:', stepIndex);
        const newBotMessage = document.createElement('div');
        newBotMessage.className = 'message bot';
        newBotMessage.dataset.stepIndex = stepIndex;
        newBotMessage.innerText = responseText || 'Response not available';
        conversation.appendChild(newBotMessage);
        conversation.scrollTop = conversation.scrollHeight;
    }
}

function populateChatWindow(projectData) {
    console.log('Populating chat window with projectData:', projectData);
    const conversation = document.getElementById('conversation');
    conversation.innerHTML = '';  // Clear existing content

    projectData.steps.forEach((step, stepIndex) => {
        console.log('Processing stepIndex:', stepIndex);

        // Insert user's input
        const userInput = step.step_config.input_text;
        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        userMessage.dataset.stepIndex = stepIndex;
        userMessage.innerText = userInput;
        conversation.appendChild(userMessage);

        // Insert a single randomly chosen response
        const responses = step.responses;
        const randomResponse = responses[Math.floor(Math.random() * responses.length)].response_content;
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot';
        botMessage.dataset.stepIndex = stepIndex;
        botMessage.innerText = randomResponse;
        conversation.appendChild(botMessage);

        console.log('Added userMessage and botMessage for stepIndex:', stepIndex);
    });

    document.getElementById('displayed-experiment-name').innerText = projectData.name;

    // Use the latest step's data for initial visualizations
    initializeVisualizations(projectData.steps[0].umap_data.embeddings, projectData.steps[0].responses);

    if (projectData.steps.length > 0) {
        const lastStepIndex = projectData.steps.length - 1;
        highlightMessage(lastStepIndex);
        updateVisualizations(lastStepIndex, projectData);
    }
    
    // Add click event listeners for messages
    document.querySelectorAll('.message').forEach(message => {
        message.addEventListener('click', function () {
            const stepIndex = parseInt(this.dataset.stepIndex);
            console.log('Message clicked for stepIndex:', stepIndex);
            updateVisualizations(stepIndex, projectData);
            highlightMessage(stepIndex);
        });
    });
}


function highlightMessage(stepIndex) {
    console.log(`Highlighting message for stepIndex: ${stepIndex}`);
    document.querySelectorAll('.message').forEach(message => {
        message.classList.remove('highlight');
        if (parseInt(message.dataset.stepIndex) === stepIndex) {
            message.classList.add('highlight');
        }
    });
}



function updateVisualizations(stepIndex, projectData) {
    const step = projectData.steps[stepIndex];
    if (step && step.umap_data) {
        const embeddings = step.umap_data.embeddings;
        const responses = step.responses;
        const temperatures = responses.map(response => response.configuration.temperature);
        createUMAPVisualization(embeddings, responses);
        createGridVisualization(embeddings, responses, temperatures);
    }
}

