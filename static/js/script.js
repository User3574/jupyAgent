// Highlight.js for syntax highlighting
hljs.highlightAll();

// Global counter
var globalCounter = 0;
var globalStep = Null;

// Re-render LaTeX after new content is added
function renderLatex() {
  MathJax.typeset();
}

function toggleGroup(header) {
  const group = header.parentElement;
  group.classList.toggle('collapsed');
}

function handleSubmit() {
    var code = document.getElementById('pythonCode').value.trim();  // Get the value of the textarea

    if (code) {
        // Clear the textarea after submission
        document.getElementById('pythonCode').value = "";

        // Update Global Step
        globalCounter++;
        globalStep = `history-${globalCounter}`;

        // Set default empty History and Setup for the Agent Responses
        var template = `
<div id="interaction-cell">
    <h2>User Prompt #${globalCounter}</h2>
    <p>${code}</p>
    <div class="group">
        <div class="group-header" onclick="toggleGroup(this)">History</div>
        <div id="${globalStep}" class="group-content"></div>
    </div>
</div>
`;
        $(".notebook").append(template);

        // Send the data to Flask using AJAX POST
        $.ajax({
            url: '/submit',  // The Flask route that handles code submission
            method: 'POST',
            data: { code: code },  // Send the code as form data
            success: function(response) {
                console.log("Response from server:", response);

                // Start the streaming connection after the code is successfully received
                startSSEStream();
            },
            error: function(error) {
                console.error('Error:', error);
                // Optionally handle the error here
            }
        });
    } else {
        console.log("No code to submit!");  // In case there's no input
    }
}

// Function to handle the SSE stream
function startSSEStream() {
    const eventSource = new EventSource('/stream');  // Use the /stream route for SSE
    var localCounter = 0;
    var localStep = `history-${globalCounter}-step-${localCounter}`;

    // Handle incoming messages from the server
    eventSource.onmessage = function(event) {
        // Append the received data to the notebook
        const data = JSON.parse(event.data);
        const msg = data.msg;
        const newStep = data.new_step;

        if (newStep) {
            localCounter++;
            localStep = `history-${globalCounter}-step-${localCounter}`;

            // Create a new group for the new step, using localCounter for the step label
            document.getElementById(globalStep).insertAdjacentHTML(
                'beforeend',
                `<div class="group">
                    <div class="group-header" onclick="toggleGroup(this)">Step ${localCounter}</div>
                    <div id="${localStep}" class="group-content"></div>
                </div>`
            );
        }
        document.getElementById(localStep).insertAdjacentHTML('beforeend', msg);

        // Re-render LaTeX and syntax highlighting after new content is added
        renderLatex();
        hljs.highlightAll();
    };

    // Handle errors (optional)
    eventSource.onerror = function(event) {
        console.error("Error occurred during SSE stream:", event);
        eventSource.close();  // Close the connection if an error occurs
    };
}

// Re-render LaTeX after initial page load
window.onload = renderLatex;
