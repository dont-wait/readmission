import { useState } from 'react';
import predictionService from '../services/predictionService';
import type { PatientFeatures, PredictionResponse } from '../services/predictionService';

export const usePrediction = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<PredictionResponse | null>(null);

    const getPrediction = async (patient: PatientFeatures, threshold?: number) => {
        setLoading(true);
        setError(null);
        try {
            const data = await predictionService.predict(patient, threshold);
            setResult(data);
            return data;
        } catch (err: any) {
            const errMsg = err instanceof Error ? err.message : 'Failed to get prediction';
            setError(errMsg);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    return {
        loading,
        error,
        result,
        getPrediction,
        setResult,
    };
};
