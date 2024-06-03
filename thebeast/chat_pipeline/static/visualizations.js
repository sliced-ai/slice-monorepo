function createTSNEVisualization(embeddings, responses) {
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
            const index = embeddings.indexOf(d);
            console.log('UMAP point clicked:', d, 'index:', index, 'response:', responses[index]);
            updateChatWindow(responses[index]);
            highlightUMAPPoint(index);
            highlightGridPlane(index);
        });

    function highlightUMAPPoint(index) {
        svg.selectAll("circle").classed("highlight", false);
        svg.selectAll("circle").filter((d, i) => i === index).classed("highlight", true);
    }

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
    console.log('Creating 3D visualization with embeddings:', embeddings, 'and responses:', responses);

    // Add a constant third dimension to the embeddings
    const embeddings3D = embeddings.map(d => [d[0], d[1], 0]); // Third dimension is currently 0

    const trace = {
        x: embeddings3D.map(d => d[0]),
        y: embeddings3D.map(d => d[1]),
        z: embeddings3D.map(d => d[2]),
        mode: 'markers',
        type: 'scatter3d',
        text: responses,
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
                console.log('Grid plane clicked:', index);
                const responseText = responses[index] || 'Response not available';
                updateChatWindow(responseText);
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
