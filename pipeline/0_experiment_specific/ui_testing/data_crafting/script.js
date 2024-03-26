document.addEventListener('DOMContentLoaded', function() {
    const chatDisplay = document.getElementById('chatDisplay');
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');

    function updateChatDisplay(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        messageElement.textContent = `${sender === 'user' ? 'You' : 'AI'}: ${message}`;
        chatDisplay.appendChild(messageElement);
        chatDisplay.scrollTop = chatDisplay.scrollHeight; // Scroll to the bottom
    }

    function sendChatToServer(chatText) {
        const requestUrl = '/generate_text_datacraft';

        updateChatDisplay('user', chatText); // Show user's message in the chat

        fetch(requestUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ paragraph: chatText }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const serverResponse = data.generated_text || "No response";
            updateChatDisplay('ai', serverResponse); // Show AI's response in the chat
        })
        .catch((error) => {
            updateChatDisplay('error', `Error: ${error.message}`);
        });
    }

    sendButton.addEventListener('click', function(event) {
        event.preventDefault();
        const chatText = chatInput.value.trim();

        if (chatText) {
            sendChatToServer(chatText);
            chatInput.value = ''; // Clear the input area
        } else {
            alert('Please enter some text to chat.');
        }
    });

    // Allow pressing "Enter" to send a message
    chatInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendButton.click();
        }
    });
});
