import React from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface InputFieldProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label: string;
    unit?: string;
    error?: string;
    icon?: React.ReactNode;
}

export const InputField: React.FC<InputFieldProps> = ({
    label,
    unit,
    error,
    icon,
    className,
    ...props
}) => {
    return (
        <div className={`flex flex-col p-6 bg-white border border-slate-200 rounded-2xl shadow-sm hover:shadow-md transition-all ${className}`}>
            <div className="flex items-center gap-3 mb-4">
                {icon && (
                    <div className="text-[#0047AB]">
                        {React.cloneElement(icon as any, { size: 16 })}
                    </div>
                )}
                <span className="text-[10px] font-black tracking-[0.2em] text-slate-400 uppercase">
                    {label}
                </span>
            </div>

            <div className="flex items-center justify-between mb-3">
                <input
                    className="w-full text-2xl font-black text-slate-800 bg-transparent"
                    {...props}
                />
                <div className="flex flex-col text-slate-300">
                    <ChevronUp size={14} className="cursor-pointer hover:text-[#0047AB] transition-all" />
                    <ChevronDown size={14} className="cursor-pointer hover:text-[#0047AB] transition-all" />
                </div>
            </div>

            {unit && (
                <div className="text-[10px] text-slate-400 font-black uppercase tracking-widest mt-auto">
                    {unit}
                </div>
            )}

            {error && <p className="text-[10px] text-red-600 mt-1 font-bold">{error}</p>}
        </div>
    );
};
