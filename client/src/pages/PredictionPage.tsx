import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm } from 'react-hook-form';
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
    LineChart,
    Pill
} from 'lucide-react';
import { usePrediction } from '../hooks/usePrediction';
import type { PatientFeatures } from '../services/predictionService';
import predictionService from '../services/predictionService';
import { InputField } from '../components/common/InputField';
import { Button } from '../components/common/Button';

const formatMetric = (val: unknown): string => {
    if (typeof val === 'number') return val >= 0 && val <= 1 ? `${(val * 100).toFixed(1)}%` : val.toFixed(3);
    return String(val);
};

const getClinicalStrategy = (probability: number, threshold: number, label: number): { tier: string; text: string } => {
    if (label === 1) {
        const margin = (probability - threshold) / threshold;
        if (margin > 0.5) return {
            tier: 'CRITICAL',
            text: `CRITICAL ALERT: Readmission probability ${(probability * 100).toFixed(1)}% — ${((probability - threshold) * 100).toFixed(1)}pp above threshold. Immediate hospitalization evaluation and multi-disciplinary intervention required.`,
        };
        return {
            tier: 'HIGH RISK',
            text: `URGENT PRECAUTION: High intervention priority. Immediate comprehensive clinical review of discharge metrics and heart failure management protocol is strongly mandated.`,
        };
    }
    const nearThreshold = probability > threshold * 0.75;
    if (nearThreshold) return {
        tier: 'BORDERLINE',
        text: `CLOSE MONITORING: Probability ${(probability * 100).toFixed(1)}% is within ${((threshold - probability) * 100).toFixed(1)}pp of threshold. Intensify follow-up schedule and reassess within 7 days.`,
    };
    return {
        tier: 'STABLE',
        text: `STANDARD OBSERVATION: Patient stability metrics fall within safe readmission threshold. Continue with standard heart failure follow-up protocols.`,
    };
};

const PredictionPage: React.FC = () => {
    const { getPrediction, loading, error, result } = usePrediction();
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('');

    const { register, handleSubmit, formState: { errors } } = useForm<PatientFeatures>({
        defaultValues: {
            age: 70,
            bmi: 28.1,
            bnp: 456,
            sodium: 137.5,
            creatinine: 1.2,
            systolic_bp: 130,
            heart_rate: 82,
            ace_inhibitor: 1,
            beta_blocker: 1,
            diuretic: 0,
            adherence_score: 0.62,
            distance_to_hospital_km: 24.5,
        },
    });

    useEffect(() => {
        predictionService.getHealth().then((h) => {
            if (h.available_models.length > 0) {
                h.available_models.shift();
            }
            setAvailableModels(h.available_models);
            setSelectedModel(h.available_models[0]);
        }).catch(() => { });
    }, []);

    const onSubmit = async (data: PatientFeatures) => {
        try {
            await getPrediction(data, undefined, selectedModel || undefined);
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
                    <form onSubmit={handleSubmit(onSubmit)} className="p-8 space-y-16">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                            <InputField
                                label="AGE"
                                type="number"
                                icon={<User />}
                                unit="Years"
                                error={errors.age?.message}
                                {...register('age', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 18, message: 'Min 18' },
                                    max: { value: 120, message: 'Max 120' },
                                })}
                            />
                            <InputField
                                label="BMI"
                                type="number"
                                step="0.1"
                                icon={<Scale />}
                                unit="kg/m²"
                                error={errors.bmi?.message}
                                {...register('bmi', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 10, message: 'Min 10' },
                                    max: { value: 60, message: 'Max 60' },
                                })}
                            />
                            <InputField
                                label="BNP LEVEL"
                                type="number"
                                icon={<Activity />}
                                unit="pg/mL"
                                error={errors.bnp?.message}
                                {...register('bnp', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 0, message: 'Min 0' },
                                    max: { value: 5000, message: 'Max 5000' },
                                })}
                            />
                            <InputField
                                label="SODIUM"
                                type="number"
                                step="0.1"
                                icon={<Droplets />}
                                unit="mEq/L"
                                error={errors.sodium?.message}
                                {...register('sodium', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 100, message: 'Min 100' },
                                    max: { value: 170, message: 'Max 170' },
                                })}
                            />
                            <InputField
                                label="CREATININE"
                                type="number"
                                step="0.1"
                                icon={<FlaskConical />}
                                unit="mg/dL"
                                error={errors.creatinine?.message}
                                {...register('creatinine', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 0.1, message: 'Min 0.1' },
                                    max: { value: 20, message: 'Max 20' },
                                })}
                            />
                            <InputField
                                label="SYSTOLIC BP"
                                type="number"
                                icon={<Activity />}
                                unit="mmHg"
                                error={errors.systolic_bp?.message}
                                {...register('systolic_bp', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 60, message: 'Min 60' },
                                    max: { value: 250, message: 'Max 250' },
                                })}
                            />
                            <InputField
                                label="HEART RATE"
                                type="number"
                                icon={<Heart />}
                                unit="BPM"
                                error={errors.heart_rate?.message}
                                {...register('heart_rate', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 20, message: 'Min 20' },
                                    max: { value: 250, message: 'Max 250' },
                                })}
                            />
                            <InputField
                                label="ACE INHIBITOR"
                                type="number"
                                step="1"
                                icon={<Pill />}
                                unit="0 / 1"
                                error={errors.ace_inhibitor?.message}
                                {...register('ace_inhibitor', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 0, message: 'Must be 0 or 1' },
                                    max: { value: 1, message: 'Must be 0 or 1' },
                                })}
                            />
                            <InputField
                                label="BETA BLOCKER"
                                type="number"
                                step="1"
                                icon={<Pill />}
                                unit="0 / 1"
                                error={errors.beta_blocker?.message}
                                {...register('beta_blocker', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 0, message: 'Must be 0 or 1' },
                                    max: { value: 1, message: 'Must be 0 or 1' },
                                })}
                            />
                            <InputField
                                label="DIURETIC"
                                type="number"
                                step="1"
                                icon={<Pill />}
                                unit="0 / 1"
                                error={errors.diuretic?.message}
                                {...register('diuretic', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 0, message: 'Must be 0 or 1' },
                                    max: { value: 1, message: 'Must be 0 or 1' },
                                })}
                            />
                            <InputField
                                label="ADHERENCE SCORE"
                                type="number"
                                step="0.01"
                                icon={<ClipboardCheck />}
                                unit="0 - 100%"
                                error={errors.adherence_score?.message}
                                {...register('adherence_score', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 0, message: 'Min 0' },
                                    max: { value: 1, message: 'Max 1' },
                                })}
                            />
                            <InputField
                                label="HOSPITAL DISTANCE"
                                type="number"
                                step="0.1"
                                icon={<MapPin />}
                                unit="Miles"
                                error={errors.distance_to_hospital_km?.message}
                                {...register('distance_to_hospital_km', {
                                    valueAsNumber: true,
                                    required: 'Required',
                                    min: { value: 0, message: 'Min 0' },
                                    max: { value: 1000, message: 'Max 1000' },
                                })}
                            />
                        </div>

                        {availableModels.length > 0 && (
                            <div className="space-y-4">
                                <p className="mt-4 text-xs font-black text-slate-400 uppercase tracking-[0.3em]">Prediction Model</p>
                                <div className="mt-4 flex flex-wrap gap-3">
                                    {availableModels.map((m) => (
                                        <button
                                            key={m}
                                            type="button"
                                            onClick={() => setSelectedModel(m)}
                                            className={`px-5 py-2.5 rounded-xl text-xs font-black uppercase tracking-[0.15em] border transition-all ${selectedModel === m
                                                ? 'bg-[#004282] text-white border-[#004282] shadow-lg shadow-blue-900/20'
                                                : 'bg-white text-slate-500 border-slate-200 hover:border-[#004282] hover:text-[#004282]'
                                                }`}
                                        >
                                            {m}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

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
                    {result && (() => {
                        const strategy = getClinicalStrategy(result.readmission_probability, result.threshold, result.predicted_label);
                        const isHigh = result.predicted_label === 1;
                        const tierStyle: Record<string, { bg: string; border: string; badge: string; accent: string }> = {
                            'CRITICAL': { bg: 'bg-red-50/50', border: 'border-red-200', badge: 'bg-[#e63946] text-white', accent: 'bg-[#e63946]' },
                            'HIGH RISK': { bg: 'bg-orange-50/50', border: 'border-orange-200', badge: 'bg-orange-500 text-white', accent: 'bg-orange-500' },
                            'BORDERLINE': { bg: 'bg-amber-50/50', border: 'border-amber-200', badge: 'bg-amber-100 text-amber-700', accent: 'bg-amber-400' },
                            'STABLE': { bg: 'bg-emerald-50/50', border: 'border-emerald-200', badge: 'bg-[#2a9d8f] text-white', accent: 'bg-[#2a9d8f]' },
                        };
                        const ts = tierStyle[strategy.tier] ?? tierStyle['STABLE'];

                        return (
                            <motion.div
                                key="result"
                                initial={{ opacity: 0, y: 60 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: 60 }}
                                transition={{ type: 'spring', damping: 25, stiffness: 120 }}
                                className="mt-16 rounded-3xl overflow-hidden shadow-[0_40px_80px_-20px_rgba(0,0,0,0.12)]"
                            >
                                {/* Zone 1 — Dark Score Hero */}
                                <div className="relative px-10 py-12 overflow-hidden">
                                    <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-[0.07] blur-3xl pointer-events-none ${isHigh ? 'bg-[#e63946]' : 'bg-[#2a9d8f]'}`} />
                                    <div className="relative z-10 flex flex-col sm:flex-row items-start sm:items-center gap-8 sm:gap-14">
                                        <div className="flex flex-col items-center shrink-0">
                                            <p className="text-[9px] font-black text-white/25 uppercase tracking-[0.5em] mb-2">30-Day Readmission</p>
                                            <div className="flex items-end gap-1.5">
                                                <motion.span
                                                    initial={{ opacity: 0, y: 16 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: 0.2, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                                                    className={`font-black leading-none tracking-tighter ${isHigh ? 'text-[#ff4d6d]' : 'text-[#26c485]'}`}
                                                    style={{ fontSize: 'clamp(5rem, 11vw, 8rem)' }}
                                                >
                                                    {Math.round(result.readmission_probability * 100)}
                                                </motion.span>
                                                <span className="text-white/25 font-black text-3xl mb-3">%</span>
                                            </div>
                                            <span className={`inline-flex px-3.5 py-1.5 rounded-full text-[9px] font-black uppercase tracking-[0.2em] border mt-3 ${isHigh ? 'text-[#ff4d6d] bg-[#e63946]/10 border-[#e63946]/25' : 'text-[#26c485] bg-[#2a9d8f]/10 border-[#2a9d8f]/25'}`}>
                                                {isHigh ? 'High Risk' : 'Low Risk'}
                                            </span>
                                        </div>

                                        <div className="flex-1 w-full space-y-4">
                                            <div className="relative">
                                                <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                                                    <motion.div
                                                        initial={{ width: 0 }}
                                                        animate={{ width: `${result.readmission_probability * 100}%` }}
                                                        transition={{ duration: 2.2, ease: [0.22, 1, 0.36, 1], delay: 0.35 }}
                                                        className={`h-full rounded-full ${isHigh ? 'bg-gradient-to-r from-amber-400 to-[#e63946]' : 'bg-gradient-to-r from-teal-400 to-[#26c485]'}`}
                                                    />
                                                </div>
                                                <div
                                                    className="absolute top-1/2 -translate-y-1/2 w-px h-5 bg-white/30"
                                                    style={{ left: `${result.threshold * 100}%` }}
                                                />
                                            </div>
                                            <div className="flex justify-between text-[8px] font-black text-white/15 uppercase tracking-[0.3em]">
                                                <span className="mx-auto text-white/30">Threshold {(result.threshold * 100).toFixed(1)}%</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Zone 2 — Metrics + Confusion Matrix */}
                                <div className="bg-white border-t border-slate-100 p-8 space-y-8">
                                    {result.model_metrics ? (
                                        <>
                                            <div>
                                                <p className="text-[9px] font-black uppercase tracking-[0.4em] mb-4">Model Performance</p>
                                                <div className="grid grid-cols-3 gap-2">
                                                    {[
                                                        { label: 'Accuracy', value: result.model_metrics.accuracy },
                                                        { label: 'ROC AUC', value: result.model_metrics.roc_auc },
                                                        { label: 'F1 Score', value: result.model_metrics.f1 },
                                                        { label: 'Precision', value: result.model_metrics.precision },
                                                        { label: 'Recall', value: result.model_metrics.recall },
                                                        { label: 'Avg Prec', value: result.model_metrics.average_precision },
                                                    ].map((m, i) => (
                                                        <motion.div
                                                            key={m.label}
                                                            initial={{ opacity: 0, y: 8 }}
                                                            animate={{ opacity: 1, y: 0 }}
                                                            transition={{ delay: 0.05 * i, duration: 0.35 }}
                                                            className="group hover:bg-[#004282] rounded-xl px-4 py-4 border border-slate-100 hover:border-transparent transition-all duration-300 cursor-default"
                                                        >
                                                            <p className="text-[8px] mb-4 font-black uppercase tracking-[0.15em] group-hover:text-white/40 mb-1 transition-colors">{m.label}</p>
                                                            <p className="text-sm font-black text-slate-400 group-hover:text-white tracking-tight transition-colors">{formatMetric(m.value)}</p>
                                                        </motion.div>
                                                    ))}
                                                </div>
                                            </div>

                                            <div className="pt-2 border-t border-slate-100">
                                                <div className="flex items-center justify-between mb-5">
                                                    <p className="text-[9px] font-black mt-4 uppercase tracking-[0.4em]">Confusion Matrix</p>
                                                    {result.model_metrics.predicted_positive > 0 && (
                                                        <span className="text-[9px] font-bold text-slate-300 font-mono">n = {result.model_metrics.predicted_positive}</span>
                                                    )}
                                                </div>
                                                <div className="grid grid-cols-2 gap-2.5">
                                                    {[
                                                        { label: 'True Neg', value: result.model_metrics.tn, bg: 'bg-slate-50', border: 'border-slate-200', num: 'text-slate-700', sub: 'text-slate-400' },
                                                        { label: 'False Pos', value: result.model_metrics.fp, bg: 'bg-red-50/60', border: 'border-red-100', num: 'text-red-600', sub: 'text-red-300' },
                                                        { label: 'False Neg', value: result.model_metrics.fn, bg: 'bg-orange-50/60', border: 'border-orange-100', num: 'text-orange-600', sub: 'text-orange-300' },
                                                        { label: 'True Pos', value: result.model_metrics.tp, bg: 'bg-emerald-50/60', border: 'border-emerald-100', num: 'text-emerald-600', sub: 'text-emerald-300' },
                                                    ].map((cell, i) => (
                                                        <motion.div
                                                            key={cell.label}
                                                            initial={{ opacity: 0, scale: 0.93 }}
                                                            animate={{ opacity: 1, scale: 1 }}
                                                            transition={{ delay: 0.3 + 0.07 * i, duration: 0.35 }}
                                                            className={`${cell.bg} border ${cell.border} rounded-2xl px-4 py-5 text-center`}
                                                        >
                                                            <p className={`text-[8px] font-black uppercase tracking-[0.2em] ${cell.sub} mb-2`}>{cell.label}</p>
                                                            <p className={`font-black ${cell.num} leading-none tabular-nums`} style={{ fontSize: 'clamp(1.75rem, 4vw, 2.5rem)' }}>
                                                                {cell.value > 0 ? cell.value : '—'}
                                                            </p>
                                                        </motion.div>
                                                    ))}
                                                </div>
                                            </div>
                                        </>
                                    ) : (
                                        <div className="py-10 text-center bg-slate-50 rounded-2xl border border-dashed border-slate-200">
                                            <p className="text-xs font-bold text-slate-400 italic">Extended model telemetry unavailable</p>
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        );
                    })()}
                </AnimatePresence>
            </main>
        </div>
    );
};

export default PredictionPage;
