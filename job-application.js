// Job application form handling
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing job application form...');
    
    // Get elements
    const elements = {
        applyButton: document.getElementById('applyNowButton'),
        modal: document.getElementById('jobApplicationModal'),
        form: document.getElementById('applicationForm'),
        coverLetter: document.getElementById('cover_letter'),
        charCount: document.getElementById('charCount'),
        submitBtn: document.getElementById('submitApplication'),
        alertContainer: document.querySelector('.alert-container'),
        loader: document.getElementById('loadingOverlay'),
        validationMessage: document.getElementById('validation-message')
    };
    
    // Debug logging
    console.log('Elements found:', {
        applyButton: !!elements.applyButton,
        modal: !!elements.modal,
        form: !!elements.form,
        coverLetter: !!elements.coverLetter,
        charCount: !!elements.charCount,
        submitBtn: !!elements.submitBtn,
        alertContainer: !!elements.alertContainer,
        loader: !!elements.loader,
        validationMessage: !!elements.validationMessage
    });

    let isSubmitting = false;

    if (elements.form) {
        elements.form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (isSubmitting) {
                console.log('Form submission already in progress');
                return;
            }
            
            console.log('Form submission started');
            console.log('Form data:', {
                coverLetter: elements.coverLetter?.value?.length || 0,
                action: this.action,
                method: this.method
            });

            // Validate required fields
            const formData = new FormData(this);
            const validationErrors = [];
            
            // Cover letter validation
            const coverLetter = formData.get('cover_letter')?.trim();
            console.log('Cover letter validation:', {
                length: coverLetter?.length,
                isEmpty: !coverLetter,
                tooShort: coverLetter?.length < 100,
                tooLong: coverLetter?.length > 5000
            });

            if (!coverLetter) {
                validationErrors.push('Cover letter is required');
            } else if (coverLetter.length < 100) {
                validationErrors.push('Cover letter must be at least 100 characters');
            } else if (coverLetter.length > 5000) {
                validationErrors.push('Cover letter cannot exceed 5000 characters');
            }

            if (validationErrors.length > 0) {
                console.warn('Validation errors:', validationErrors);
                showError(validationErrors.join('\n'));
                return;
            }

            isSubmitting = true;

            // Show loading state
            updateUI('submitting');
            console.log('Form action URL:', this.action);

            try {
                console.log('Sending application data...');
                
                const response = await fetch(this.action, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    },
                    credentials: 'same-origin'
                });

                console.log('Response status:', response.status);
                const data = await response.json();
                console.log('Server response:', data);

                if (!response.ok) {
                    let errorMessage;
                    switch (data.error) {
                        case 'CONNECTION_ERROR':
                            errorMessage = 'Database connection error. Please try again later.';
                            break;
                        case 'VALIDATION_ERROR':
                            errorMessage = data.message || 'Please check your application details.';
                            break;
                        case 'DUPLICATE_APPLICATION':
                            errorMessage = 'You have already applied for this job.';
                            break;
                        case 'SCHEMA_ERROR':
                            errorMessage = 'Application data is invalid. Please try again.';
                            break;
                        default:
                            errorMessage = data.message || 'An unexpected error occurred.';
                    }
                    throw new Error(errorMessage);
                }

                await showSuccess(data.message || 'Application submitted successfully!');
                // Reset form and close modal
                this.reset();
                const modal = bootstrap.Modal.getInstance(elements.modal);
                if (modal) {
                    modal.hide();
                }

                // Redirect if URL provided
                if (data.redirectUrl) {
                    window.location.href = data.redirectUrl;
                }
            } catch (error) {
                console.error('Submission error:', error);
                showError(error.message);
            } finally {
                isSubmitting = false;
                updateUI('idle');
            }
        });
    }

    // Cover letter validation
    if (elements.coverLetter && elements.charCount && elements.submitBtn) {
        const debouncedValidate = debounce((value) => {
            const length = value.trim().length;
            console.log('Cover letter length updated:', length);
            
            elements.charCount.textContent = length;
            const isValid = length >= 100 && length <= 5000;
            
            elements.submitBtn.disabled = !isValid;
            elements.coverLetter.classList.toggle('is-valid', isValid);
            elements.coverLetter.classList.toggle('is-invalid', !isValid);
            
            if (elements.validationMessage) {
                if (!isValid) {
                    elements.validationMessage.textContent = length < 100 
                        ? `Please add ${100 - length} more characters` 
                        : `Please remove ${length - 5000} characters`;
                    elements.validationMessage.classList.remove('text-success');
                    elements.validationMessage.classList.add('text-danger');
                } else {
                    elements.validationMessage.textContent = 'Cover letter length is good!';
                    elements.validationMessage.classList.remove('text-danger');
                    elements.validationMessage.classList.add('text-success');
                }
            }
        }, 300);

        elements.coverLetter.addEventListener('input', function() {
            debouncedValidate(this.value);
        });
    }

    // Helper functions
    function showError(message) {
        console.error('Error:', message);
        Swal.fire({
            title: 'Error!',
            text: message,
            icon: 'error',
            confirmButtonText: 'OK'
        });
    }

    function showSuccess(message) {
        return Swal.fire({
            title: 'Success!',
            text: message,
            icon: 'success',
            confirmButtonText: 'OK'
        });
    }

    function updateUI(state) {
        if (state === 'submitting') {
            elements.submitBtn.disabled = true;
            elements.submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Submitting...';
            if (elements.loader) {
                elements.loader.classList.add('active');
            }
        } else {
            elements.submitBtn.disabled = false;
            elements.submitBtn.innerHTML = 'Submit Application';
            if (elements.loader) {
                elements.loader.classList.remove('active');
            }
        }
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
});