function createUMAPVisualization(embeddings, responses) {
    console.log('Creating UMAP visualization with embeddings:', embeddings, 'and responses:', responses);
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
        .text("UMAP Visualization");

    svg.selectAll("circle")
        .data(embeddings)
        .enter().append("circle")
        .attr("cx", d => x(d[0]))
        .attr("cy", d => y(d[1]))
        .attr("r", 5)
        .style("fill", "orange")
        .on("click", function(event, d) {
            const umapIndex = embeddings.indexOf(d);
            const response = responses[umapIndex].response_content;
            console.log(`UMAP point clicked: UMAP Index: ${umapIndex}, Response: ${response}`);
            
            // Update the chat window with the response from UMAP point
            const chatWindowIndex = document.querySelector('.message.highlight')?.dataset.stepIndex;
            if (chatWindowIndex !== undefined) {
                updateChatWindow(response, chatWindowIndex);
            } else {
                console.warn('No chat window index selected to update');
            }
        });
}



function createGridVisualization(embeddings, responses) {
    console.log('Creating 3D visualization with embeddings:', embeddings, 'and responses:', responses);

    // Add a constant third dimension to the embeddings
    const embeddings3D = embeddings.map(d => [d[0], d[1], 0]); // Third dimension is currently 0

    const trace = {
        x: embeddings3D.map(d => d[0]),
        y: embeddings3D.map(d => d[1]),
        z: embeddings3D.map(d => d[2]),
        mode: 'markers',
        type: 'scatter3d',
        text: responses.map(r => r.response_content),
        marker: {
            size: 5,
            color: 'orange'
        }
    };

    const layout = {
        title: '3D UMAP Visualization',
        scene: {
            xaxis: {title: 'UMAP 1'},
            yaxis: {title: 'UMAP 2'},
            zaxis: {title: 'UMAP 3'},
        }
    };

    Plotly.newPlot('grid-visual', [trace], layout).then(() => {
        console.log('Grid visualization rendered successfully.');

        const gridPlot = document.getElementById('grid-visual');
        gridPlot.on('plotly_click', function(data) {
            if (data.points.length > 0) {
                const index = data.points[0].pointIndex;
                console.log(`Grid plane clicked at index: ${index}`);
                const responseText = responses[index].response_content || 'Response not available';
                updateChatWindow(responseText, index);
                highlight3DPoint(index);
            }
        });

    }).catch(error => {
        console.error('Failed to render grid visualization:', error);
    });

    function highlight3DPoint(index) {
        const update = {
            marker: {
                color: embeddings3D.map((_, i) => i === index ? 'blue' : 'orange')
            }
        };
        Plotly.restyle('grid-visual', update);
    }
}

