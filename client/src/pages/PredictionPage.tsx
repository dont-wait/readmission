import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Activity,
    User,
    Heart,
    MapPin,
    Scale,
    AlertCircle,
    FlaskConical,
    Droplets,
    ClipboardCheck,
    LineChart
} from 'lucide-react';
import { usePrediction } from '../hooks/usePrediction';
import type { PatientFeatures } from '../services/predictionService';
import { InputField } from '../components/common/InputField';
import { Button } from '../components/common/Button';

const PredictionPage: React.FC = () => {
    const { getPrediction, loading, error, result } = usePrediction();
    const [formData, setFormData] = useState<PatientFeatures>({
        age: 70,
        bmi: 28.1,
        bnp: 456,
        sodium: 137.5,
        creatinine: 1.2,
        systolic_bp: 130,
        heart_rate: 82,
        adherence_score: 0.62,
        distance_to_hospital_km: 24.5,
    });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData((prev) => ({
            ...prev,
            [name]: parseFloat(value) || 0,
        }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            await getPrediction(formData);
        } catch (err) {
            console.error('Prediction failed:', err);
        }
    };

    return (
        <div className="min-h-screen bg-[#fbfcff] text-slate-900 font-sans pb-20">
            {/* Header */}
            <header className="border-b border-slate-200 bg-white sticky top-0 z-50">
                <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="w-[1px] h-6 bg-slate-200" />
                        <h1 className="text-lg font-bold text-[#001f3f] tracking-tight">Readmission Predictor</h1>
                    </div>
                    <div className="w-10 h-10 rounded-full bg-[#0047AB] shadow-sm flex items-center justify-center">
                        <div className="w-8 h-8 rounded-full bg-white/20" />
                    </div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 pt-24 pb-16">
                <div className="mb-16">
                    <h2 className="text-5xl font-bold text-[#011627] mb-5 tracking-tight">Vitals Entry</h2>
                    <p className="text-slate-500 font-medium text-xl opacity-70">Input current clinical metrics to calculate the probability of 30-day readmission.</p>
                </div>

                <div className="bg-white rounded-[40px] border border-slate-100 shadow-[0_32px_64px_-16px_rgba(0,0,0,0.06)]">
                    <form onSubmit={handleSubmit} className="p-8 space-y-16">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                            <InputField
                                label="AGE"
                                name="age"
                                type="number"
                                value={formData.age}
                                onChange={handleChange}
                                icon={<User />}
                                unit="Years"
                            />
                            <InputField
                                label="BMI"
                                name="bmi"
                                type="number"
                                step="0.1"
                                value={formData.bmi}
                                onChange={handleChange}
                                icon={<Scale />}
                                unit="kg/m²"
                            />
                            <InputField
                                label="BNP LEVEL"
                                name="bnp"
                                type="number"
                                value={formData.bnp}
                                onChange={handleChange}
                                icon={<Activity />}
                                unit="pg/mL"
                            />
                            <InputField
                                label="SODIUM"
                                name="sodium"
                                type="number"
                                step="0.1"
                                value={formData.sodium}
                                onChange={handleChange}
                                icon={<Droplets />}
                                unit="mEq/L"
                            />
                            <InputField
                                label="CREATININE"
                                name="creatinine"
                                type="number"
                                step="0.1"
                                value={formData.creatinine}
                                onChange={handleChange}
                                icon={<FlaskConical />}
                                unit="mg/dL"
                            />
                            <InputField
                                label="SYSTOLIC BP"
                                name="systolic_bp"
                                type="number"
                                value={formData.systolic_bp}
                                onChange={handleChange}
                                icon={<Activity />}
                                unit="mmHg"
                            />
                            <InputField
                                label="HEART RATE"
                                name="heart_rate"
                                type="number"
                                value={formData.heart_rate}
                                onChange={handleChange}
                                icon={<Heart />}
                                unit="BPM"
                            />
                            <InputField
                                label="ADHERENCE SCORE"
                                name="adherence_score"
                                type="number"
                                step="0.01"
                                min="0"
                                max="1"
                                value={formData.adherence_score}
                                onChange={handleChange}
                                icon={<ClipboardCheck />}
                                unit="0 - 100%"
                            />
                            <InputField
                                label="HOSPITAL DISTANCE"
                                name="distance_to_hospital_km"
                                type="number"
                                step="0.1"
                                value={formData.distance_to_hospital_km}
                                onChange={handleChange}
                                icon={<MapPin />}
                                unit="Miles"
                            />
                        </div>

                        <div className="flex flex-col items-center gap-6">
                            <Button
                                type="submit"
                                className="mt-10 mb-10 bg-[#004282] hover:bg-[#003366] text-white px-14 py-6 rounded-2xl flex items-center gap-4 text-xl font-bold shadow-2xl shadow-blue-900/20 transition-all active:scale-[0.97]"
                                isLoading={loading}
                            >
                                <div className="p-1.5 border border-white/20 rounded-lg bg-white/10 shadow-inner">
                                    <LineChart size={24} />
                                </div>
                                Analyze Prediction
                            </Button>
                        </div>

                        {error && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="flex items-center gap-4 p-5 rounded-2xl bg-red-50 border border-red-100 text-red-600 text-sm font-bold shadow-sm"
                            >
                                <AlertCircle size={20} />
                                {error}
                            </motion.div>
                        )}
                    </form>
                </div>

                <AnimatePresence mode="wait">
                    {result && (
                        <motion.div
                            key="result"
                            initial={{ opacity: 0, y: 60 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 60 }}
                            transition={{ type: "spring", damping: 25, stiffness: 120 }}
                            className="mt-32 bg-white border border-slate-200 rounded-[48px] pt-16 shadow-[0_40px_80px_-20px_rgba(0,0,0,0.08)] relative overflow-hidden"
                        >
                            <div className="absolute top-0 right-0 w-80 h-80 bg-[#004282]/[0.03] rounded-full -mr-40 -mt-40" />

                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 relative z-10">
                                <div>
                                    <h3 className="text-slate-400 font-black mb-10 uppercase text-xs tracking-[0.4em]">Risk Assessment</h3>
                                    <div className="flex items-baseline gap-5 mb-12">
                                        <span className={`text-9xl font-black tracking-tighter ${(result.readmission_probability * 100) > 50 ? 'text-[#e63946]' : 'text-[#2a9d8f]'}`}>
                                            {Math.round(result.readmission_probability * 100)}%
                                        </span>
                                        <span className="text-slate-300 text-2xl font-bold tracking-tight opacity-50 uppercase">Probability</span>
                                    </div>

                                    <div className="space-y-6">
                                        <div className="flex justify-between items-center px-2">
                                            <span className="text-xs font-black text-slate-400 uppercase tracking-[0.2em]">Classification</span>
                                            <span className={`px-6 py-2.5 rounded-full text-[10px] font-black uppercase tracking-[0.2em] shadow-sm ${result.predicted_label === 1 ? 'bg-red-50 text-[#e63946] border border-red-100' : 'bg-emerald-50 text-[#2a9d8f] border border-emerald-100'}`}>
                                                {result.predicted_label === 1 ? 'High Risk' : 'Low Risk'}
                                            </span>
                                        </div>
                                        <div className="w-full bg-slate-50 h-6 rounded-full overflow-hidden p-2 border border-slate-100 shadow-inner">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${result.readmission_probability * 100}%` }}
                                                transition={{ duration: 2, ease: [0.22, 1, 0.36, 1] }}
                                                className={`h-full rounded-full ${result.predicted_label === 1 ? 'bg-[#e63946] shadow-[0_0_24px_rgba(230,57,70,0.4)]' : 'bg-[#2a9d8f] shadow-[0_0_24px_rgba(42,157,143,0.4)]'}`}
                                            />
                                        </div>
                                        <div className="flex justify-between px-3 text-[10px] font-black text-slate-300 uppercase tracking-[0.3em]">
                                            <span>Baseline Stable</span>
                                            <span>Acute Decompensation</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="flex flex-col justify-between py-2">
                                    <div className="space-y-10">
                                        <div className="p-10 rounded-[32px] border border-indigo-50 bg-indigo-50/20 shadow-sm shadow-indigo-100/10">
                                            <div className="flex gap-6">
                                                <div className="p-5 rounded-2xl bg-[#004282] text-white shadow-2xl shadow-blue-200">
                                                    <Activity size={28} />
                                                </div>
                                                <div>
                                                    <h4 className="font-black text-slate-800 text-2xl tracking-tight">Diagnostics Hub</h4>
                                                    <p className="text-sm text-slate-500 mt-2 font-bold italic opacity-60">XGBoost Statistical Confidence: {(result.threshold * 100).toFixed(1)}%</p>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="p-10 rounded-[32px] border border-slate-100 bg-slate-50/40">
                                            <h4 className="font-black text-[#001f3f] mb-5 uppercase text-[10px] tracking-[0.3em]">Clinical Strategy</h4>
                                            <p className="text-base text-slate-600 leading-relaxed font-bold opacity-90 italic">
                                                {result.predicted_label === 1
                                                    ? "URGENT PRECAUTION: High intervention priority. Immediate comprehensive clinical review of discharge metrics and heart failure management protocol is strongly mandated."
                                                    : "STANDARD OBSERVATION: Patient stability metrics fall within safe readmission threshold. Continue with standard heart failure follow-up protocols."}
                                            </p>
                                        </div>
                                    </div>

                                    <button
                                        className="w-full mt-12 bg-[#011627] text-white py-6 rounded-2xl font-black flex items-center justify-center gap-4 hover:bg-black transition-all shadow-2xl shadow-slate-200 active:scale-[0.98] uppercase tracking-[0.3em] text-[10px]"
                                        onClick={() => window.print()}
                                    >
                                        Extract Clinical Analysis Report
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </main>
        </div>
    );
};

export default PredictionPage;
