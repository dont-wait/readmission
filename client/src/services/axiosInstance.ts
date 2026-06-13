import axios from 'axios';

const axiosInstance = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    headers: {
        'Content-Type': 'application/json',
    },
});

// Response interceptor for error handling
axiosInstance.interceptors.response.use(
    (response) => response.data,
    (error) => {
        const message = error.response?.data?.detail || error.message || 'An unexpected error occurred';
        // You can add global error toast notification here
        return Promise.reject(new Error(message));
    }
);

export default axiosInstance;
