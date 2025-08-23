// static/dashboard.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const connectionStatusLight = document.getElementById('status-light');
    const connectionStatusText = document.getElementById('status-text');
    const logFeed = document.getElementById('log-feed');
    const modal = document.getElementById('job-details-modal');
    const closeModalButton = document.getElementById('close-modal-button');
    const modalJobId = document.getElementById('modal-job-id');
    const modalDetailsContent = document.getElementById('modal-details-content');

    const jobLists = {
        pending: document.getElementById('pending-list'),
        processing: document.getElementById('processing-list'),
        completed: document.getElementById('completed-list'),
        failed: document.getElementById('failed-list'),
    };

    const jobCounts = {
        pending: document.getElementById('pending-count'),
        processing: document.getElementById('processing-count'),
        completed: document.getElementById('completed-count'),
        failed: document.getElementById('failed-count'),
    };

    // Store all job data in memory for quick access
    let allJobsData = {};
    let socket;

    // --- WebSocket Connection Handling ---
    function connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/dashboard/ws/dashboard`;

        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            console.log('WebSocket connection established.');
            updateConnectionStatus(true);
            // Request initial data dump upon connection
            socket.send(JSON.stringify({ action: 'get_all_jobs' }));
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };

        socket.onclose = () => {
            console.log('WebSocket connection closed. Attempting to reconnect in 3 seconds...');
            updateConnectionStatus(false);
            setTimeout(connectWebSocket, 3000); // Retry connection after 3 seconds
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus(false);
            socket.close();
        };
    }

    function updateConnectionStatus(isConnected) {
        if (isConnected) {
            connectionStatusLight.className = 'status-light connected';
            connectionStatusText.textContent = 'Connected';
        } else {
            connectionStatusLight.className = 'status-light disconnected';
            connectionStatusText.textContent = 'Disconnected';
        }
    }


    // --- WebSocket Message Processing ---
    function handleWebSocketMessage(data) {
        switch (data.type) {
            case 'log':
                appendLogLine(data.payload);
                break;
            case 'all_jobs':
                // This is a full state update, so clear and re-render everything
                allJobsData = {};
                data.payload.forEach(job => {
                    allJobsData[job.job_id] = job;
                });
                renderAllJobs();
                break;
            // Future enhancement: Handle single job updates for efficiency
            // case 'job_update':
            //     allJobsData[data.payload.job_id] = data.payload;
            //     renderAllJobs(); // Simple re-render for now
            //     break;
        }
    }


    // --- UI Rendering ---
    function renderAllJobs() {
        // Clear all current lists
        Object.values(jobLists).forEach(list => list.innerHTML = '');
        Object.values(jobCounts).forEach(countEl => countEl.textContent = '0');

        const counts = { pending: 0, processing: 0, completed: 0, failed: 0 };

        // Sort jobs by submission date (newest first)
        const sortedJobs = Object.values(allJobsData).sort(
            (a, b) => new Date(b.submitted_at) - new Date(a.submitted_at)
        );

        // Populate lists with sorted jobs
        sortedJobs.forEach(job => {
            const status = job.status;
            if (jobLists[status]) {
                const jobCard = createJobCard(job);
                jobLists[status].appendChild(jobCard);
                counts[status]++;
            }
        });

        // Update counts in the UI
        Object.keys(counts).forEach(status => {
            jobCounts[status].textContent = counts[status];
        });
    }

    function createJobCard(job) {
        const card = document.createElement('div');
        card.className = `job-card ${job.status}`;
        card.dataset.jobId = job.job_id;

        const description = job.message || 'No description provided.';
        const submittedTime = new Date(job.submitted_at).toLocaleString();

        card.innerHTML = `
            <div class="job-id">${job.job_id.split('-')[0]}...</div>
            <div class="job-description">${description.substring(0, 60)}...</div>
            <div class="job-timestamp">Submitted: ${submittedTime}</div>
        `;

        card.addEventListener('click', () => showJobDetails(job.job_id));
        return card;
    }

    function appendLogLine(logMessage) {
        const logLine = document.createElement('div');
        logLine.className = 'log-line';

        // Add color coding based on log level
        if (logMessage.includes('ERROR') || logMessage.includes('CRITICAL')) {
            logLine.classList.add('ERROR');
        } else if (logMessage.includes('WARNING')) {
            logLine.classList.add('WARNING');
        } else {
            logLine.classList.add('INFO');
        }

        logLine.textContent = logMessage;
        logFeed.appendChild(logLine);

        // Auto-scroll to the bottom
        logFeed.scrollTop = logFeed.scrollHeight;
    }


    // --- Modal Handling ---
    function showJobDetails(jobId) {
        const job = allJobsData[jobId];
        if (!job) return;

        modalJobId.textContent = `Job Details: ${jobId}`;
        
        // Use a <pre> tag for nicely formatted JSON
        const contentHtml = `<pre>${JSON.stringify(job, null, 2)}</pre>`;
        modalDetailsContent.innerHTML = contentHtml;

        modal.style.display = 'block';
    }

    function hideModal() {
        modal.style.display = 'none';
    }

    closeModalButton.addEventListener('click', hideModal);
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            hideModal();
        }
    });

    // --- Initial Kick-off ---
    connectWebSocket();
});