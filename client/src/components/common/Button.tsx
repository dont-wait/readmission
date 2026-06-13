import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'outline' | 'danger';
    isLoading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
    children,
    className,
    variant = 'primary',
    isLoading,
    disabled,
    ...props
}) => {
    const variants = {
        primary: 'bg-[#004282] text-white hover:bg-[#003366] shadow-xl shadow-blue-900/20',
        secondary: 'bg-[#1e293b] text-white hover:bg-black shadow-lg shadow-slate-200',
        outline: 'border border-slate-200 bg-transparent text-slate-600 hover:bg-slate-50',
        danger: 'bg-red-600 text-white hover:bg-red-700 shadow-lg shadow-red-200',
    };

    return (
        <button
            className={cn(
                'px-8 py-4 rounded-2xl font-black transition-all active:scale-[0.97] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 uppercase tracking-widest text-[10px]',
                variants[variant],
                className
            )}
            disabled={disabled || isLoading}
            {...props}
        >
            {isLoading && (
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            )}
            {children}
        </button>
    );
};
