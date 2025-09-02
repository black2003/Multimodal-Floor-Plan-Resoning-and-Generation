import React, { createContext, useContext, useState, useCallback } from 'react';

export interface AnalysisResult {
  id: string;
  timestamp: string;
  image: string;
  query?: string;
  vqa?: {
    answer: string;
    confidence: number;
    top_answers: Array<{ text: string; probability: number }>;
  };
  retrieval?: {
    similar_plans: Array<{ id: number; similarity: number; description: string }>;
    similarities: number[];
  };
  validity?: {
    validity_score: number;
    issues: any;
    suggestions: any;
    constraint_scores: Record<string, number>;
  };
  metadata: {
    image_shape: [number, number];
    analysis_type: string;
    text_query?: string;
    segmentation_info: {
      num_rooms: number;
      num_doors: number;
      num_walls: number;
    };
  };
}

export interface ExplainabilityResult {
  image?: any;
  text?: any;
  graph?: any;
  constraints?: any;
  cross_modal?: any;
  summary?: any;
}

type AnalysisContextType = {
  currentAnalysis: AnalysisResult | null;
  analysisHistory: AnalysisResult[];
  explainabilityResults: Record<string, ExplainabilityResult>;
  isLoading: boolean;
  error: string | null;
  setCurrentAnalysis: (analysis: AnalysisResult | null) => void;
  addAnalysisToHistory: (analysis: AnalysisResult) => void;
  setExplainabilityResults: (id: string, results: ExplainabilityResult) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearHistory: () => void;
};

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([]);
  const [explainabilityResults, setExplainabilityResultsState] = useState<Record<string, ExplainabilityResult>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addAnalysisToHistory = useCallback((analysis: AnalysisResult) => {
    setAnalysisHistory(prev => [analysis, ...prev.slice(0, 49)]); // Keep last 50 analyses
  }, []);

  const setExplainabilityResults = useCallback((id: string, results: ExplainabilityResult) => {
    setExplainabilityResultsState(prev => ({
      ...prev,
      [id]: results
    }));
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    setIsLoading(loading);
  }, []);

  const clearHistory = useCallback(() => {
    setAnalysisHistory([]);
    setExplainabilityResultsState({});
  }, []);

  const value: AnalysisContextType = {
    currentAnalysis,
    analysisHistory,
    explainabilityResults,
    isLoading,
    error,
    setCurrentAnalysis,
    addAnalysisToHistory,
    setExplainabilityResults,
    setLoading,
    setError,
    clearHistory,
  };

  return (
    <AnalysisContext.Provider value={value}>
      {children}
    </AnalysisContext.Provider>
  );
};
