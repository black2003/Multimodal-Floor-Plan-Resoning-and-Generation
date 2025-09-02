import React, { useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Progress } from '../components/ui/progress';
import { useToast } from '../hooks/use-toast';
import { useAnalysis } from '../contexts/AnalysisContext';
import { useSocket } from '../contexts/SocketContext';
import { 
  Upload, 
  MessageSquare, 
  Search, 
  CheckCircle, 
  Brain,
  Image as ImageIcon,
  FileText,
  BarChart3,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { ImageUploader } from '../components/ImageUploader';
import { QueryInput } from '../components/QueryInput';
import { ResultsDisplay } from '../components/ResultsDisplay';
import { VisualizationPanel } from '../components/VisualizationPanel';

const AnalysisPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const { toast } = useToast();
  const { 
    currentAnalysis, 
    setCurrentAnalysis, 
    addAnalysisToHistory, 
    isLoading, 
    setLoading, 
    setError 
  } = useAnalysis();
  const { socket, isConnected } = useSocket();
  
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [analysisType, setAnalysisType] = useState(searchParams.get('type') || 'full');
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');

  const analysisTypes = [
    { id: 'full', label: 'Full Analysis', icon: Brain, description: 'Complete multi-modal analysis' },
    { id: 'vqa', label: 'VQA', icon: MessageSquare, description: 'Visual Question Answering' },
    { id: 'retrieval', label: 'Retrieval', icon: Search, description: 'Similarity search' },
    { id: 'validity', label: 'Validity', icon: CheckCircle, description: 'Constraint validation' },
  ];

  const handleImageUpload = useCallback((imageData: string) => {
    setUploadedImage(imageData);
    setCurrentAnalysis(null);
    toast({
      title: "Image uploaded successfully",
      description: "Floor plan image is ready for analysis",
    });
  }, [setCurrentAnalysis, toast]);

  const handleQueryChange = useCallback((newQuery: string) => {
    setQuery(newQuery);
  }, []);

  const handleAnalysisTypeChange = useCallback((type: string) => {
    setAnalysisType(type);
  }, []);

  const startAnalysis = useCallback(async () => {
    if (!uploadedImage) {
      toast({
        title: "No image uploaded",
        description: "Please upload a floor plan image first",
        variant: "destructive",
      });
      return;
    }

    if (analysisType === 'vqa' && !query.trim()) {
      toast({
        title: "No query provided",
        description: "Please enter a question for VQA analysis",
        variant: "destructive",
      });
      return;
    }

    if (!isConnected) {
      toast({
        title: "Connection error",
        description: "Please check your connection to the server",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    setProgress(0);
    setError(null);

    try {
      // Prepare analysis data
      const analysisData = {
        image: uploadedImage,
        query: query.trim(),
        type: analysisType,
        timestamp: new Date().toISOString(),
      };

      // Emit analysis request via WebSocket
      socket?.emit('analyze_floorplan', analysisData);

      // Listen for progress updates
      socket?.on('analysis_progress', (data: any) => {
        setCurrentStep(data.step || '');
        setProgress(data.progress || 0);
      });

      // Listen for results
      socket?.on('analysis_result', (result: any) => {
        const analysisResult = {
          id: `analysis_${Date.now()}`,
          timestamp: new Date().toISOString(),
          image: uploadedImage,
          query: query.trim(),
          ...result,
          metadata: {
            ...result.metadata,
            analysis_type: analysisType,
          }
        };

        setCurrentAnalysis(analysisResult);
        addAnalysisToHistory(analysisResult);
        setLoading(false);
        setProgress(100);

        toast({
          title: "Analysis completed",
          description: "Floor plan analysis has been completed successfully",
        });
      });

      // Listen for errors
      socket?.on('analysis_error', (error: any) => {
        setError(error.error || 'Analysis failed');
        setLoading(false);
        setProgress(0);

        toast({
          title: "Analysis failed",
          description: error.error || 'An error occurred during analysis',
          variant: "destructive",
        });
      });

    } catch (error) {
      setError('Failed to start analysis');
      setLoading(false);
      setProgress(0);

      toast({
        title: "Analysis failed",
        description: "Failed to start analysis. Please try again.",
        variant: "destructive",
      });
    }
  }, [uploadedImage, query, analysisType, isConnected, socket, setLoading, setError, setCurrentAnalysis, addAnalysisToHistory, toast]);

  const getAnalysisTypeInfo = (type: string) => {
    return analysisTypes.find(t => t.id === type) || analysisTypes[0];
  };

  const currentTypeInfo = getAnalysisTypeInfo(analysisType);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Floor Plan Analysis</h1>
            <p className="text-muted-foreground">
              Upload floor plan images and analyze them using multi-modal AI
            </p>
          </div>
          <Badge variant="outline" className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Panel - Input */}
        <div className="lg:col-span-1 space-y-6">
          {/* Analysis Type Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="h-5 w-5" />
                <span>Analysis Type</span>
              </CardTitle>
              <CardDescription>
                Choose the type of analysis to perform
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {analysisTypes.map((type) => {
                const Icon = type.icon;
                return (
                  <Button
                    key={type.id}
                    variant={analysisType === type.id ? 'default' : 'outline'}
                    className="w-full justify-start h-auto p-4"
                    onClick={() => handleAnalysisTypeChange(type.id)}
                  >
                    <div className="flex items-center space-x-3">
                      <Icon className="h-5 w-5" />
                      <div className="text-left">
                        <div className="font-medium">{type.label}</div>
                        <div className="text-xs opacity-70">{type.description}</div>
                      </div>
                    </div>
                  </Button>
                );
              })}
            </CardContent>
          </Card>

          {/* Image Upload */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <ImageIcon className="h-5 w-5" />
                <span>Upload Floor Plan</span>
              </CardTitle>
              <CardDescription>
                Upload a floor plan image for analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ImageUploader onImageUpload={handleImageUpload} />
            </CardContent>
          </Card>

          {/* Query Input (for VQA) */}
          {analysisType === 'vqa' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <MessageSquare className="h-5 w-5" />
                  <span>Question</span>
                </CardTitle>
                <CardDescription>
                  Ask a question about the floor plan
                </CardDescription>
              </CardHeader>
              <CardContent>
                <QueryInput 
                  value={query}
                  onChange={handleQueryChange}
                  placeholder="e.g., How many bedrooms are in this floor plan?"
                />
              </CardContent>
            </Card>
          )}

          {/* Analysis Button */}
          <Card>
            <CardContent className="pt-6">
              <Button 
                onClick={startAnalysis}
                disabled={!uploadedImage || isLoading || (analysisType === 'vqa' && !query.trim())}
                className="w-full"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="h-5 w-5 mr-2" />
                    Start {currentTypeInfo.label}
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Progress */}
          {isLoading && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Analysis Progress</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Progress value={progress} className="w-full" />
                <p className="text-sm text-muted-foreground">{currentStep}</p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="lg:col-span-2 space-y-6">
          {currentAnalysis ? (
            <Tabs defaultValue="results" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="results">Results</TabsTrigger>
                <TabsTrigger value="visualization">Visualization</TabsTrigger>
                <TabsTrigger value="explainability">Explainability</TabsTrigger>
              </TabsList>
              
              <TabsContent value="results" className="space-y-4">
                <ResultsDisplay analysis={currentAnalysis} />
              </TabsContent>
              
              <TabsContent value="visualization" className="space-y-4">
                <VisualizationPanel analysis={currentAnalysis} />
              </TabsContent>
              
              <TabsContent value="explainability" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Explainability</CardTitle>
                    <CardDescription>
                      Detailed explanations of the analysis results
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center py-8 text-muted-foreground">
                      <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Explainability features coming soon...</p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <div className="text-center space-y-4">
                  <div className="p-4 rounded-full bg-muted">
                    <FileText className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">No Analysis Results</h3>
                    <p className="text-muted-foreground">
                      Upload a floor plan image and start analysis to see results here
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;
