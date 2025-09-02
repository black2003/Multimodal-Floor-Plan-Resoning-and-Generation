import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  MessageSquare, 
  Search, 
  CheckCircle, 
  Brain,
  TrendingUp,
  AlertTriangle,
  Info
} from 'lucide-react';
import { AnalysisResult } from '../contexts/AnalysisContext';

interface ResultsDisplayProps {
  analysis: AnalysisResult;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ analysis }) => {
  const renderVQAResults = () => {
    if (!analysis.vqa) return null;

    const { answer, confidence, top_answers } = analysis.vqa;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <MessageSquare className="h-5 w-5" />
            <span>Visual Question Answering</span>
          </CardTitle>
          <CardDescription>
            Answer to: "{analysis.query}"
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-medium">Answer:</span>
              <Badge variant="outline" className="flex items-center space-x-1">
                <TrendingUp className="h-3 w-3" />
                <span>{(confidence * 100).toFixed(1)}% confidence</span>
              </Badge>
            </div>
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-lg font-semibold">{answer}</p>
            </div>
          </div>

          {top_answers && top_answers.length > 1 && (
            <div className="space-y-2">
              <span className="text-sm font-medium">Alternative answers:</span>
              <div className="space-y-1">
                {top_answers.slice(1, 4).map((alt, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-background rounded border">
                    <span className="text-sm">{alt.text}</span>
                    <Badge variant="secondary" className="text-xs">
                      {(alt.probability * 100).toFixed(1)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderRetrievalResults = () => {
    if (!analysis.retrieval) return null;

    const { similar_plans, similarities } = analysis.retrieval;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Search className="h-5 w-5" />
            <span>Similar Floor Plans</span>
          </CardTitle>
          <CardDescription>
            Found {similar_plans.length} similar floor plans
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3">
            {similar_plans.map((plan, index) => (
              <div key={plan.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                <div className="space-y-1">
                  <p className="font-medium">Plan #{plan.id}</p>
                  <p className="text-sm text-muted-foreground">{plan.description}</p>
                </div>
                <div className="flex items-center space-x-2">
                  <Progress value={plan.similarity * 100} className="w-20" />
                  <Badge variant="outline">
                    {(plan.similarity * 100).toFixed(1)}%
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderValidityResults = () => {
    if (!analysis.validity) return null;

    const { validity_score, issues, suggestions, constraint_scores } = analysis.validity;

    const getScoreColor = (score: number) => {
      if (score >= 0.8) return 'text-green-600';
      if (score >= 0.6) return 'text-yellow-600';
      return 'text-red-600';
    };

    const getScoreBadge = (score: number) => {
      if (score >= 0.8) return 'default';
      if (score >= 0.6) return 'secondary';
      return 'destructive';
    };

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5" />
            <span>Floor Plan Validity</span>
          </CardTitle>
          <CardDescription>
            Structural validation and constraint checking
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Overall Score */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-medium">Overall Validity Score</span>
              <Badge variant={getScoreBadge(validity_score)}>
                {(validity_score * 100).toFixed(1)}%
              </Badge>
            </div>
            <Progress value={validity_score * 100} className="w-full" />
          </div>

          {/* Constraint Scores */}
          <div className="space-y-3">
            <span className="font-medium">Constraint Analysis</span>
            <div className="grid gap-2">
              {Object.entries(constraint_scores).map(([constraint, score]) => (
                <div key={constraint} className="flex items-center justify-between">
                  <span className="text-sm capitalize">
                    {constraint.replace('_', ' ')}
                  </span>
                  <div className="flex items-center space-x-2">
                    <Progress value={score * 100} className="w-16" />
                    <span className={`text-sm font-medium ${getScoreColor(score)}`}>
                      {(score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Issues */}
          {issues && Object.keys(issues).length > 0 && (
            <div className="space-y-2">
              <span className="font-medium flex items-center space-x-2">
                <AlertTriangle className="h-4 w-4 text-yellow-600" />
                <span>Issues Found</span>
              </span>
              <div className="space-y-2">
                {Object.entries(issues).map(([key, issue]: [string, any]) => (
                  <div key={key} className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                    <p className="text-sm font-medium">{issue.type || key}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {issue.description || 'Issue detected'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Suggestions */}
          {suggestions && Object.keys(suggestions).length > 0 && (
            <div className="space-y-2">
              <span className="font-medium flex items-center space-x-2">
                <Info className="h-4 w-4 text-blue-600" />
                <span>Repair Suggestions</span>
              </span>
              <div className="space-y-2">
                {Object.entries(suggestions).map(([key, suggestion]: [string, any]) => (
                  <div key={key} className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                    <p className="text-sm font-medium">{suggestion.type || key}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {suggestion.description || 'Suggestion available'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderMetadata = () => {
    const { metadata } = analysis;
    const { segmentation_info } = metadata;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>Analysis Metadata</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <span className="text-sm font-medium">Image Size</span>
              <p className="text-sm text-muted-foreground">
                {metadata.image_shape[0]} × {metadata.image_shape[1]} pixels
              </p>
            </div>
            <div className="space-y-2">
              <span className="text-sm font-medium">Analysis Type</span>
              <Badge variant="outline">{metadata.analysis_type}</Badge>
            </div>
            <div className="space-y-2">
              <span className="text-sm font-medium">Rooms Detected</span>
              <p className="text-sm text-muted-foreground">{segmentation_info.num_rooms}</p>
            </div>
            <div className="space-y-2">
              <span className="text-sm font-medium">Doors Detected</span>
              <p className="text-sm text-muted-foreground">{segmentation_info.num_doors}</p>
            </div>
            <div className="space-y-2">
              <span className="text-sm font-medium">Walls Detected</span>
              <p className="text-sm text-muted-foreground">{segmentation_info.num_walls}</p>
            </div>
            <div className="space-y-2">
              <span className="text-sm font-medium">Timestamp</span>
              <p className="text-sm text-muted-foreground">
                {new Date(analysis.timestamp).toLocaleString()}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      {analysis.vqa && renderVQAResults()}
      {analysis.retrieval && renderRetrievalResults()}
      {analysis.validity && renderValidityResults()}
      {renderMetadata()}
    </div>
  );
};
