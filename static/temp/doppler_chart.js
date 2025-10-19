// Read data from the embedded JSON script
const chartDataElement = document.getElementById('chart-data');
const chartData = JSON.parse(chartDataElement.textContent);

const xData = chartData.xData;
const yData = chartData.yData;

// Create the chart
const ctx = document.getElementById('signalPlot').getContext('2d');
new Chart(ctx, {
    type: 'line',
    data: {
        labels: xData,
        datasets: [{
            label: 'Doppler-Shifted Signal',
            data: yData,
            borderColor: 'rgba(0, 234, 255, 0.9)',
            backgroundColor: 'rgba(0, 234, 255, 0.1)',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true,
            tension: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            intersect: false,
            mode: 'index'
        },
        scales: {
            x: {
                type: 'linear',
                title: { 
                    display: true, 
                    text: 'Time (seconds)', 
                    color: '#00eaff',
                    font: { size: 14, weight: 'bold' }
                },
                ticks: { 
                    color: '#e0e0e0',
                    maxTicksLimit: 10
                },
                grid: { color: '#333' }
            },
            y: {
                title: { 
                    display: true, 
                    text: 'Amplitude', 
                    color: '#00eaff',
                    font: { size: 14, weight: 'bold' }
                },
                ticks: { 
                    color: '#e0e0e0',
                    callback: function(value) {
                        return value.toFixed(2);
                    }
                },
                grid: { color: '#333' }
            }
        },
        plugins: {
            legend: { 
                labels: { 
                    color: '#e0e0e0',
                    font: { size: 12 }
                } 
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: '#00eaff',
                bodyColor: '#e0e0e0',
                callbacks: {
                    label: function(context) {
                        return 'Amplitude: ' + context.parsed.y.toFixed(4);
                    },
                    title: function(context) {
                        return 'Time: ' + context[0].parsed.x.toFixed(4) + 's';
                    }
                }
            }
        }
    }
});