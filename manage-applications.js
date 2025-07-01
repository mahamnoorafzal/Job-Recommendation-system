// Handle application status updates
function updateApplicationStatus(applicationId, status) {
    if (!confirm(`Are you sure you want to ${status} this application?`)) {
        return;
    }
    
    fetch(`/update_application_status/${applicationId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({ status: status })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })    .then(data => {
        if (data.success) {  // Changed from data.status to data.success to match server response
            // Refresh the page to show updated status
            location.reload();
        } else {
            throw new Error(data.error || data.message || 'Unknown error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error updating application status: ' + error.message);
    });
}
