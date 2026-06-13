import axiosInstance from './axiosInstance';

export interface PatientFeatures {
    age: number;
    bmi: number;
    bnp: number;
    sodium: number;
    creatinine: number;
    systolic_bp: number;
    heart_rate: number;
    ace_inhibitor: number;
    beta_blocker: number;
    diuretic: number;
    adherence_score: number;
    distance_to_hospital_km: number;
}

export interface ModelMetrics {
    accuracy: number;
    roc_auc: number;
    f1: number;
    precision: number;
    recall: number;
    average_precision: number;
    tn: number;
    fp: number;
    fn: number;
    tp: number;
    predicted_positive: number;
}

export interface PredictionResponse {
    readmission_probability: number;
    predicted_label: number;
    threshold: number;
    model_id: string;
    model_path: string;
    model_metrics: ModelMetrics | null;
    model_metrics_path: string | null;
}

export interface HealthResponse {
    status: string;
    active_model_id: string;
    active_model_path: string;
    threshold: number;
    feature_columns: string[];
    available_models: string[];
    expects_preprocessed_features: boolean;
}

const predictionService = {
    getHealth: async (): Promise<HealthResponse> => {
        return axiosInstance.get('/health');
    },

    getFeatures: async (): Promise<{ feature_columns: string[] }> => {
        return axiosInstance.get('/features');
    },

    predict: async (patient: PatientFeatures, threshold?: number, model?: string): Promise<PredictionResponse> => {
        const params: Record<string, unknown> = {};
        if (threshold !== undefined) params.threshold = threshold;
        if (model !== undefined) params.model = model;
        return axiosInstance.post('/predict', patient, { params });
    },
};

export default predictionService;
