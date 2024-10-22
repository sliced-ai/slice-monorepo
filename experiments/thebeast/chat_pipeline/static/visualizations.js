function createUMAPVisualization(embeddings, responses) {
    console.log('Creating UMAP visualization with embeddings:', embeddings, 'and responses:', responses);
    const svg = d3.select("#tsne-visual");
    svg.selectAll("*").remove();
    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;

    const x = d3.scaleLinear()
        .domain([d3.min(embeddings, d => d[0]), d3.max(embeddings, d => d[0])])
        .range([50, width - 50]);

    const y = d3.scaleLinear()
        .domain([d3.min(embeddings, d => d[1]), d3.max(embeddings, d => d[1])])
        .range([height - 50, 50]);

    svg.append("g")
        .attr("transform", `translate(0,${height - 50})`)
        .call(d3.axisBottom(x));

    svg.append("g")
        .attr("transform", "translate(50,0)")
        .call(d3.axisLeft(y));

    svg.append("text")
        .attr("x", width / 2)
        .attr("y", 30)
        .attr("text-anchor", "middle")
        .style("font-size", "20px")
        .style("text-decoration", "underline")

    svg.selectAll("circle")
        .data(embeddings)
        .enter().append("circle")
        .attr("cx", d => x(d[0]))
        .attr("cy", d => y(d[1]))
        .attr("r", 5)
        .style("fill", d3.scaleOrdinal(d3.schemeCategory10)) // Using d3 color scheme for better distinction
        .on("click", function(event, d) {
            const umapIndex = embeddings.indexOf(d);
            const response = responses[umapIndex].response_content;
            console.log('UMAP point clicked:', `UMAP Index: ${umapIndex}`, `Response: ${response}`);
            const chatIndex = document.querySelector('.message.bot.highlight')?.dataset.stepIndex;
            if (chatIndex !== undefined) {
                updateChatWindow(response, chatIndex);
                highlightUMAPPoint(umapIndex);
            } else {
                console.log('No chat window index selected to update');
            }
        });

    function highlightUMAPPoint(index) {
        svg.selectAll("circle").classed("highlight", false);
        svg.selectAll("circle").filter((d, i) => i === index).classed("highlight", true);
    }
}

let currentEmbeddings = [];
let currentResponses = [];

document.getElementById('z-axis-select').addEventListener('change', function() {
    const selectedAxis = this.value;
    createGridVisualization(currentEmbeddings, currentResponses, selectedAxis);
});

function initializeVisualizations(embeddings, responses) {
    currentEmbeddings = embeddings;
    currentResponses = responses;
    createUMAPVisualization(embeddings, responses);
    createGridVisualization(embeddings, responses);
}

function createGridVisualization(embeddings, responses, zAxisParameter = 'temperature') {
    console.log('Creating 3D visualization with embeddings:', embeddings, 'and responses:', responses, 'using z-axis parameter:', zAxisParameter);

    if (responses.length === 0) {
        console.error('No responses available for visualization');
        return;
    }

    // Ensure zAxisParameter is a valid string
    if (typeof zAxisParameter !== 'string') {
        console.error('Invalid zAxisParameter:', zAxisParameter);
        zAxisParameter = 'temperature';  // Default to 'temperature' if invalid
    }

    const zValues = responses.map(r => r.configuration[zAxisParameter]);
    const embeddings3D = embeddings.map((d, i) => [d[0], d[1], zValues[i]]);

    const trace = {
        x: embeddings3D.map(d => d[0]),
        y: embeddings3D.map(d => d[1]),
        z: embeddings3D.map(d => d[2]),
        mode: 'markers',
        type: 'scatter3d',
        text: responses.map(r => r.response_content),
        marker: {
            size: 5,
            color: zValues,
            colorscale: 'Viridis',
        },
        hovertemplate: '<b>' + zAxisParameter.charAt(0).toUpperCase() + zAxisParameter.slice(1) + ':</b> %{z:.2f}<br>' +
                       '<b>UMAP 1:</b> %{x:.2f}<br>' +
                       '<b>UMAP 2:</b> %{y:.2f}<br>' +
                       '<b>Response:</b> %{text}<extra></extra>'
    };

    const layout = {
        scene: {
            xaxis: {title: 'UMAP 1'},
            yaxis: {title: 'UMAP 2'},
            zaxis: {title: zAxisParameter.charAt(0).toUpperCase() + zAxisParameter.slice(1)},
        },
        width: 800,
        height: 600,
        margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 50,
        },
        hovermode: 'closest',
    };

    Plotly.newPlot('grid-visual', [trace], layout).then(() => {
        console.log('Grid visualization rendered successfully.');

        const gridPlot = document.getElementById('grid-visual');
        gridPlot.on('plotly_click', function(data) {
            console.log('Plotly click event data:', data);
            if (data && data.points && data.points.length > 0) {
                const point = data.points[0];
                console.log('Clicked point data:', point);
                const index = point.pointNumber;  // This should be the correct property
                console.log('Index from Plotly click event:', index);
                const response = responses[index];
                if (response) {
                    console.log('Grid plane clicked at index:', index, 'with response:', response.response_content);
                    const responseText = response.response_content || 'Response not available';
                    const chatIndex = document.querySelector('.message.bot.highlight')?.dataset.stepIndex;
                    if (chatIndex !== undefined) {
                        updateChatWindow(responseText, chatIndex);
                    } else {
                        console.log('No chat window index selected to update');
                    }
                } else {
                    console.error('No response found for index:', index);
                }
            } else {
                console.error('Invalid Plotly click event data:', data);
            }
        });

    }).catch(error => {
        console.error('Failed to render grid visualization:', error);
    });
}
