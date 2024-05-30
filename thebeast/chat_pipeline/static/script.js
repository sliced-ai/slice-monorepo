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

    // Add event listener for the Projects button
    document.getElementById('projects-button').addEventListener('click', function() {
        window.location.href = '/projects';
    });
});

document.getElementById('home-button').addEventListener('click', function() {
    window.location.href = '/';
});

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

    // Show user message
    const conversation = document.getElementById('conversation');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.innerText = inputText;
    conversation.appendChild(userMessage);

    // Show loading indicator
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'message bot';
    loadingIndicator.id = 'loading-indicator';
    loadingIndicator.innerText = 'Processing...';
    conversation.appendChild(loadingIndicator);

    document.getElementById('input_text').value = '';
    conversation.scrollTop = conversation.scrollHeight;

    // Set a long timeout for the fetch request
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000000); // 100 minutes

    fetch('/chat', {
        method: 'POST',
        body: formData,
        signal: controller.signal
    })
    .then(response => response.json())
    .then(data => {
        clearTimeout(timeoutId); // Clear the timeout
        console.log('Received chat response:', data);
        if (data.error) {
            alert(data.error);
            conversation.removeChild(loadingIndicator);
            return;
        }

        const botMessage = document.createElement('div');
        botMessage.className = 'message bot';
        botMessage.id = 'bot-message';  // Assign an ID for easy replacement
        botMessage.innerText = data.chosen_response; // Display only the chosen response
        conversation.replaceChild(botMessage, loadingIndicator);

        console.log('Fetching TSNE data from path:', data.tsne_data_path);
        console.log('Fetching Grid data from path:', data.grid_data_path);

        fetch(data.tsne_data_path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(tsneData => {
                console.log('Received TSNE data:', tsneData);
                if (!tsneData.embeddings || !Array.isArray(tsneData.embeddings) || tsneData.embeddings.length === 0) {
                    throw new Error('Invalid TSNE data received.');
                }
                createTSNEVisualization(tsneData.embeddings, data.responses);
            })
            .catch(error => {
                console.error('Error fetching or processing TSNE data:', error);
                alert('Failed to fetch or process TSNE data.');
            });

        fetch(data.grid_data_path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(gridData => {
                console.log('Received Grid data:', gridData);
                if (!gridData.embeddings || !Array.isArray(gridData.embeddings) || gridData.embeddings.length === 0) {
                    throw new Error('Invalid Grid data received.');
                }
                createGridVisualization(gridData.embeddings, data.responses);
            })
            .catch(error => {
                console.error('Error fetching or processing Grid data:', error);
                alert('Failed to fetch or process Grid data.');
            });

        conversation.scrollTop = conversation.scrollHeight;
    })
    .catch(error => {
        clearTimeout(timeoutId); // Clear the timeout
        console.error('Error sending chat message:', error);
        alert('Failed to send message.');
        conversation.removeChild(loadingIndicator);
    });

    // Periodically fetch and display the status
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

// Function to toggle between views
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

// Handle dark mode button
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

// Update displayed experiment name when the form input changes
document.getElementById('experiment_name').addEventListener('input', function() {
    const experimentName = this.value.trim() || 'none';
    document.getElementById('displayed-experiment-name').innerText = experimentName;
});

// Update chat window with new responses
function updateChatWindow(responseText) {
    const conversation = document.getElementById('conversation');
    const lastBotMessage = Array.from(conversation.getElementsByClassName('message bot')).pop();
    if (lastBotMessage) {
        lastBotMessage.innerText = responseText || 'Response not available';
    } else {
        const newBotMessage = document.createElement('div');
        newBotMessage.className = 'message bot';
        newBotMessage.id = 'bot-message';
        newBotMessage.innerText = responseText || 'Response not available';
        conversation.appendChild(newBotMessage);
        conversation.scrollTop = conversation.scrollHeight;
    }
}