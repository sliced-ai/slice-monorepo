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

    fetch('/chat', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Received chat response:', data);
        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById('response').innerText = data.message;
        const conversation = document.getElementById('conversation');
        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        userMessage.innerText = inputText;
        conversation.appendChild(userMessage);

        const botMessage = document.createElement('div');
        botMessage.className = 'message bot';
        botMessage.innerText = data.chosen_response;
        conversation.appendChild(botMessage);

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

        document.getElementById('input_text').value = '';
        conversation.scrollTop = conversation.scrollHeight;
    })
    .catch(error => {
        console.error('Error sending chat message:', error);
        alert('Failed to send message.');
    });
});

function createTSNEVisualization(embeddings, responses) {
    console.log('Creating TSNE visualization with embeddings:', embeddings, 'and responses:', responses);
    const svg = d3.select("#tsne-visual");
    svg.selectAll("*").remove();
    const width = +svg.attr("width");
    const height = +svg.attr("height");

    const x = d3.scaleLinear()
        .domain([d3.min(embeddings, d => d[0]), d3.max(embeddings, d => d[0])])
        .range([50, width - 50]);

    const y = d3.scaleLinear()
        .domain([d3.min(embeddings, d => d[1]), d3.max(embeddings, d => d[1])])
        .range([height - 50, 50]);

    // Add X axis
    svg.append("g")
        .attr("transform", `translate(0,${height - 50})`)
        .call(d3.axisBottom(x));

    // Add Y axis
    svg.append("g")
        .attr("transform", "translate(50,0)")
        .call(d3.axisLeft(y));

    // Add Title
    svg.append("text")
        .attr("x", width / 2)
        .attr("y", 30)
        .attr("text-anchor", "middle")
        .style("font-size", "20px")
        .style("text-decoration", "underline")
        .text("t-SNE Visualization");

    svg.selectAll("circle")
        .data(embeddings)
        .enter().append("circle")
        .attr("cx", d => x(d[0]))
        .attr("cy", d => y(d[1]))
        .attr("r", 5)
        .style("fill", "orange")
        .on("click", function(event, d) {
            const index = embeddings.indexOf(d);
            console.log('TSNE point clicked:', d, 'index:', index, 'response:', responses[index]);
            updateChatWindow(responses[index]);
            highlightTSNEPoint(index);
            highlightGridPlane(index);
        });

    function highlightTSNEPoint(index) {
        svg.selectAll("circle").classed("highlight", false);
        svg.selectAll("circle").filter((d, i) => i === index).classed("highlight", true);
    }

    // Make sure to define highlightGridPlane
    function highlightGridPlane(index) {
        const gridPlot = document.getElementById('grid-visual');
        if (gridPlot && gridPlot.data && Array.isArray(gridPlot.data[index].z)) {
            Plotly.restyle(gridPlot, 'surfacecolor', gridPlot.data.map((d, i) => {
                return i === index ? d.z.map(row => row.map(() => 'blue')) : null;
            }));
            console.log(`Highlighting plane for response ${index + 1}`);
        } else {
            console.error('Invalid grid data or gridPlot object:', gridPlot);
        }
    }
}
function createGridVisualization(embeddings, responses) {
    console.log('Received grid data for visualization:', embeddings);
    console.log('Associated responses:', responses);

    // Verify that the data is in the expected format
    if (!embeddings.every(e => Array.isArray(e) && e.every(row => Array.isArray(row)))) {
        console.error('Grid data is not in the expected 2D array format:', embeddings);
        return; // Exit if data is incorrectly formatted
    }

    // Log each grid and its dimensions
    embeddings.forEach((grid, index) => {
        console.log(`Grid ${index + 1} dimensions: ${grid.length}x${grid[0].length}`);
        console.log(`Grid ${index + 1} data:`, grid);
    });

    const data = embeddings.map((grid, i) => {
        return {
            z: grid,
            type: 'surface',
            name: `Response ${i + 1}`,
            showscale: false,
            text: responses[i],
            hoverinfo: 'text',
            hovertemplate: `<b>Response:</b> ${responses[i]}<extra></extra>`
        };
    });

    const layout = {
        title: '3D Visualization of Smoothed Text Embeddings',
        autosize: true,
        width: 800,
        height: 600,
        scene: {
            xaxis: { title: 'Dimension 1' },
            yaxis: { title: 'Dimension 2' },
            zaxis: { title: 'Embedding Value' }
        }
    };

    // Rendering the plot
    Plotly.newPlot('grid-visual', data, layout).then(() => {
        console.log('Grid visualization rendered successfully.');
    }).catch(error => {
        console.error('Failed to render grid visualization:', error);
    });

    const gridPlot = document.getElementById('grid-visual');
    gridPlot.on('plotly_click', function(data) {
        if (data.points.length > 0) {
            const index = data.points[0].curveNumber;
            console.log('Grid plane clicked:', index);
            const responseText = responses[index] || 'Response not available';
            updateChatWindow(responseText);
            highlightGridPlane(index);
            highlightTSNEPoint(index);
        }
    });

    function highlightGridPlane(index) {
        if (gridPlot && gridPlot.data && Array.isArray(gridPlot.data[index].z)) {
            const highlightColor = 'blue';
            const originalColor = null;

            const updatedColors = gridPlot.data.map((d, i) => {
                if (i === index) {
                    return { surfacecolor: d.z.map(row => row.map(() => highlightColor)) };
                } else {
                    return { surfacecolor: originalColor };
                }
            });

            Plotly.restyle('grid-visual', updatedColors).then(() => {
                console.log(`Highlighting plane for response ${index + 1}`);
            }).catch(error => {
                console.error('Error highlighting grid plane:', error);
            });
        } else {
            console.error('Invalid grid data or gridPlot object:', gridPlot);
        }
    }

    // Make sure to define highlightTSNEPoint
    function highlightTSNEPoint(index) {
        d3.select("#tsne-visual").selectAll("circle").classed("highlight", false);
        d3.select("#tsne-visual").selectAll("circle").filter((d, i) => i === index).classed("highlight", true);
    }
}

function updateChatWindow(responseText) {
    console.log('Updating chat window with response:', responseText);
    const conversation = document.getElementById('conversation');
    const botMessage = document.createElement('div');
    botMessage.className = 'message bot';
    botMessage.innerText = responseText || 'Response not available';
    conversation.appendChild(botMessage);
    conversation.scrollTop = conversation.scrollHeight;
}
