import React from 'react';
import PredictionPage from '../pages/PredictionPage';

const AppRouter: React.FC = () => {
    // Simple router for now, we can add react-router-dom if needed
    // For this single-page app, we'll just return the PredictionPage
    return <PredictionPage />;
};

export default AppRouter;
