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
                    {result && (
                        <motion.div
                            key="result"
                            initial={{ opacity: 0, y: 60 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 60 }}
                            transition={{ type: "spring", damping: 25, stiffness: 120 }}
                            className="mt-16 bg-white border border-slate-200 rounded-3xl overflow-hidden shadow-[0_40px_80px_-20px_rgba(0,0,0,0.08)] relative"
                        >
                            <div className="absolute top-0 right-0 w-80 h-80 bg-[#004282]/[0.03] rounded-full -mr-40 -mt-40" />

                            <div className="relative z-10">
                                {/* Zone 1 — Risk Hero (full width) */}
                                <div className="px-10 pt-10 pb-8 border-b border-slate-100">
                                    <h3 className="text-slate-400 font-black mb-8 uppercase text-[10px] tracking-[0.5em]">Risk Assessment</h3>
                                    <div className="flex flex-col sm:flex-row items-center gap-8 sm:gap-12">
                                        {/* Big number */}
                                        <div className="flex flex-col items-center shrink-0">
                                            <span className={`text-[7rem] leading-none font-black tracking-tighter transition-colors ${(result.readmission_probability * 100) > (result.threshold * 100) ? 'text-[#e63946]' : 'text-[#2a9d8f]'}`}>
                                                {Math.round(result.readmission_probability * 100)}%
                                            </span>
                                            <span className="text-slate-400 text-xs font-bold tracking-widest uppercase opacity-60 mt-1">Probability Score</span>
                                            <span className="text-[10px] font-black text-slate-300 uppercase tracking-[0.3em] mt-0.5">30-Day Readmission</span>
                                        </div>

                                        {/* Progress bar + classification */}
                                        <div className="flex-1 w-full space-y-5">
                                            <div className="flex justify-between items-center">
                                                <div className="space-y-0.5">
                                                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]">Classification</span>
                                                    <p className="text-xs font-bold text-slate-400 opacity-50 italic">Decision Threshold: {(result.threshold * 100).toFixed(1)}%</p>
                                                </div>
                                                <span className={`px-5 py-2 rounded-full text-[10px] font-black uppercase tracking-[0.2em] shadow-sm ${result.predicted_label === 1 ? 'bg-red-50 text-[#e63946] border border-red-100' : 'bg-emerald-50 text-[#2a9d8f] border border-emerald-100'}`}>
                                                    {result.predicted_label === 1 ? 'High Risk' : 'Low Risk'}
                                                </span>
                                            </div>
                                            <div className="w-full bg-slate-50 h-7 rounded-full overflow-hidden p-1.5 border border-slate-100 shadow-inner">
                                                <motion.div
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${result.readmission_probability * 100}%` }}
                                                    transition={{ duration: 2.5, ease: [0.22, 1, 0.36, 1] }}
                                                    className={`h-full rounded-full relative ${result.predicted_label === 1 ? 'bg-gradient-to-r from-[#e63946] to-[#ff4d6d]' : 'bg-gradient-to-r from-[#2a9d8f] to-[#26c485]'}`}
                                                >
                                                    <div className="absolute inset-0 bg-white/20 animate-pulse pointer-events-none" />
                                                </motion.div>
                                            </div>
                                            <div className="flex justify-between px-1 text-[10px] font-black text-slate-300 uppercase tracking-[0.4em]">
                                                <span>Baseline Stable</span>
                                                <span>Acute Decompensation</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Zone 2 — Diagnostics + Clinical Protocol (balanced 2-col) */}
                                <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-slate-100">
                                    {/* Diagnostics Hub */}
                                    <div className="p-8">
                                        <div className="flex gap-4 mb-7">
                                            <div className="p-4 rounded-2xl bg-[#004282] text-white shadow-lg shadow-blue-200 shrink-0">
                                                <Activity size={22} />
                                            </div>
                                            <div className="flex flex-col justify-center">
                                                <h4 className="font-black text-slate-800 text-lg tracking-tight leading-none">Diagnostics Hub</h4>
                                                <p className="text-[10px] font-bold text-slate-400 mt-1.5 uppercase tracking-widest opacity-60">Engine: {result.model_id}</p>
                                            </div>
                                        </div>

                                        {result.model_metrics ? (
                                            <div className="space-y-6">
                                                <div className="grid grid-cols-3 gap-2">
                                                    {[
                                                        { label: 'Accuracy', value: result.model_metrics.accuracy },
                                                        { label: 'ROC AUC', value: result.model_metrics.roc_auc },
                                                        { label: 'F1 Score', value: result.model_metrics.f1 },
                                                        { label: 'Precision', value: result.model_metrics.precision },
                                                        { label: 'Recall', value: result.model_metrics.recall },
                                                        { label: 'Avg Prec', value: result.model_metrics.average_precision },
                                                    ].map((m) => (
                                                        <div key={m.label} className="bg-slate-50 rounded-xl px-3 py-2.5 border border-slate-100">
                                                            <p className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-300 mb-0.5">{m.label}</p>
                                                            <p className="text-sm font-black text-slate-800 tracking-tight">{formatMetric(m.value)}</p>
                                                        </div>
                                                    ))}
                                                </div>

                                                <div className="pt-5 border-t border-slate-100">
                                                    <div className="flex justify-between items-center mb-4">
                                                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Confusion Matrix</span>
                                                        {result.model_metrics.predicted_positive === 0 ? (
                                                            <span className="text-[10px] font-bold text-slate-300 italic">N/A</span>
                                                        ) : (
                                                            <span className="text-[10px] font-bold text-slate-300 italic">{result.model_metrics.predicted_positive}</span>
                                                        )}
                                                    </div>
                                                    <div className="grid grid-cols-2 gap-3">
                                                        <div className="space-y-3 text-center">
                                                            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                                                                <p className="text-[8px] font-black text-slate-400 uppercase mb-1">True Neg</p>
                                                                {result.model_metrics.tn === 0 ? (
                                                                    <p className="text-sm font-black text-slate-700">-</p>
                                                                ) : (
                                                                    <p className="text-sm font-black text-slate-700">{result.model_metrics.tn}</p>
                                                                )}
                                                            </div>
                                                            <div className="p-3 rounded-xl bg-red-50/30 border border-red-50 text-red-600">
                                                                <p className="text-[8px] font-black text-red-300 uppercase mb-1">False Pos</p>
                                                                {result.model_metrics.fp === 0 ? (
                                                                    <p className="text-sm font-black text-slate-700">-</p>
                                                                ) : (
                                                                    <p className="text-sm font-black">{result.model_metrics.fp}</p>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <div className="space-y-3 text-center">
                                                            <div className="p-3 rounded-xl bg-orange-50/30 border border-orange-50 text-orange-600">
                                                                <p className="text-[8px] font-black text-orange-300 uppercase mb-1">False Neg</p>
                                                                {result.model_metrics.fn === 0 ? (
                                                                    <p className="text-sm font-black text-slate-700">-</p>
                                                                ) : (
                                                                    <p className="text-sm font-black">{result.model_metrics.fn}</p>
                                                                )}
                                                            </div>
                                                            <div className="p-3 rounded-xl bg-emerald-50/30 border border-emerald-50 text-emerald-600">
                                                                <p className="text-[8px] font-black text-emerald-300 uppercase mb-1">True Pos</p>
                                                                {result.model_metrics.tp === 0 ? (
                                                                    <p className="text-sm font-black text-slate-700">-</p>
                                                                ) : (
                                                                    <p className="text-sm font-black">{result.model_metrics.tp}</p>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="py-10 text-center bg-slate-50 rounded-2xl border border-dashed border-slate-200">
                                                <p className="text-xs font-bold text-slate-400 italic">Extended model telemetry unavailable</p>
                                            </div>
                                        )}
                                    </div>

                                    {/* Clinical Strategy */}
                                    {(() => {
                                        const strategy = getClinicalStrategy(result.readmission_probability, result.threshold, result.predicted_label);
                                        const tierColors: Record<string, string> = {
                                            'CRITICAL': 'bg-red-50/30',
                                            'HIGH RISK': 'bg-orange-50/20',
                                            'BORDERLINE': 'bg-yellow-50/20',
                                            'STABLE': 'bg-emerald-50/20',
                                        };
                                        const tierBorderColors: Record<string, string> = {
                                            'CRITICAL': 'border-red-200',
                                            'HIGH RISK': 'border-orange-100',
                                            'BORDERLINE': 'border-yellow-100',
                                            'STABLE': 'border-emerald-100',
                                        };
                                        const tierBadgeColors: Record<string, string> = {
                                            'CRITICAL': 'bg-red-500 text-white',
                                            'HIGH RISK': 'bg-orange-500 text-white',
                                            'BORDERLINE': 'bg-yellow-100 text-yellow-700',
                                            'STABLE': 'bg-[#2a9d8f] text-white',
                                        };
                                        return (
                                            <div className={`p-8 flex flex-col justify-center gap-6 border-t-2 lg:border-t-0 lg:border-l-2 ${tierBorderColors[strategy.tier] ?? 'border-slate-100'} ${tierColors[strategy.tier] ?? 'bg-slate-50/20'}`}>
                                                <div className="flex items-center gap-3">
                                                    <div className="w-1 h-8 rounded-full bg-[#001f3f]/10 shrink-0" />
                                                    <div>
                                                        <h4 className="font-black text-[#001f3f] uppercase text-[10px] tracking-[0.4em] mb-1">Clinical Protocol</h4>
                                                        <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-[0.15em] ${tierBadgeColors[strategy.tier] ?? ''}`}>
                                                            {strategy.tier}
                                                        </span>
                                                    </div>
                                                </div>
                                                <p className="text-base text-slate-700 leading-relaxed font-bold opacity-80 italic">
                                                    "{strategy.text}"
                                                </p>
                                            </div>
                                        );
                                    })()}
                                </div>

                                {/* Zone 3 — Export (full width footer) */}
                                <div className="px-8 py-5 border-t border-slate-100 bg-slate-50/40">
                                    <button
                                        className="w-full bg-[#011627] text-white py-5 rounded-2xl font-black flex items-center justify-center gap-3 hover:bg-black transition-all shadow-lg active:scale-[0.98] uppercase tracking-[0.2em] text-[10px]"
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
