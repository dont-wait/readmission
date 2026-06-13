import axiosInstance from './axiosInstance';

export interface PatientFeatures {
    age: number;
    bmi: number;
    bnp: number;
    sodium: number;
    creatinine: number;
    systolic_bp: number;
    heart_rate: number;
    adherence_score: number;
    distance_to_hospital_km: number;
}

export interface PredictionResponse {
    readmission_probability: number;
    predicted_label: number;
    threshold: number;
    model_path: string;
}

export interface HealthResponse {
    status: string;
    model_path: string;
    threshold: number;
    feature_columns: string[];
}

const predictionService = {
    getHealth: async (): Promise<HealthResponse> => {
        return axiosInstance.get('/health');
    },

    getFeatures: async (): Promise<{ feature_columns: string[] }> => {
        return axiosInstance.get('/features');
    },

    predict: async (patient: PatientFeatures, threshold?: number): Promise<PredictionResponse> => {
        const params = threshold !== undefined ? { threshold } : {};
        return axiosInstance.post('/predict', patient, { params });
    },

    predictBatch: async (items: PatientFeatures[], threshold?: number): Promise<{ predictions: PredictionResponse[] }> => {
        const params = threshold !== undefined ? { threshold } : {};
        return axiosInstance.post('/predict/batch', { items }, { params });
    },
};

export default predictionService;
