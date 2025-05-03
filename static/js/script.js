document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    // Function to add a message to the chat box
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message');
        
        if (isUser) {
            messageDiv.classList.add('user');
        } else {
            messageDiv.classList.add('bot');
        }
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>${message}</p>
            </div>
        `;
        
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to send a message to the server and get a response
    async function sendMessage(message) {
        try {
            const response = await fetch('/get_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });
            
            const data = await response.json();
            return data.response;
        } catch (error) {
            console.error('Error:', error);
            return 'Maaf, terjadi kesalahan dalam memproses pesan Anda.';
        }
    }

    // Function to handle sending a message
    async function handleSendMessage() {
        const message = userInput.value.trim();
        
        if (message) {
            // Add user message to chat
            addMessage(message, true);
            
            // Clear input field
            userInput.value = '';
            
            // Show typing indicator
            const typingDiv = document.createElement('div');
            typingDiv.classList.add('chat-message', 'bot', 'typing');
            typingDiv.innerHTML = `
                <div class="message-content">
                    <p>Mengetik...</p>
                </div>
            `;
            chatBox.appendChild(typingDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // Get response from server
            const botResponse = await sendMessage(message);
            
            // Remove typing indicator
            chatBox.removeChild(typingDiv);
            
            // Add bot response to chat
            addMessage(botResponse);
        }
    }

    // Event listener for send button
    sendButton.addEventListener('click', handleSendMessage);

    // Event listener for Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    });

    // Focus on input field when page loads
    userInput.focus();
});

// Function to send a suggestion chip message
function sendSuggestion(suggestion) {
    const userInput = document.getElementById('userInput');
    userInput.value = suggestion;
    document.getElementById('sendButton').click();
}