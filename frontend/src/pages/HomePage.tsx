import React from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { 
  Brain, 
  Search, 
  BarChart3, 
  FileText, 
  Upload, 
  MessageSquare,
  CheckCircle,
  ArrowRight,
  Zap
} from 'lucide-react';

const HomePage: React.FC = () => {
  const features = [
    {
      icon: Brain,
      title: 'Multi-Modal AI',
      description: 'Advanced Vision Transformer, DistilBERT, and Graph Neural Networks working together for comprehensive floor plan understanding.',
    },
    {
      icon: Search,
      title: 'Visual Question Answering',
      description: 'Ask natural language questions about floor plans and get intelligent answers with confidence scores.',
    },
    {
      icon: BarChart3,
      title: 'Similarity Retrieval',
      description: 'Find similar floor plans using contrastive learning and advanced embedding techniques.',
    },
    {
      icon: CheckCircle,
      title: 'Validity Checking',
      description: 'Neuro-symbolic validation with architectural constraint checking and repair suggestions.',
    },
    {
      icon: FileText,
      title: 'Explainable AI',
      description: 'Comprehensive explainability with Grad-CAM, attention visualization, and constraint tracing.',
    },
    {
      icon: Zap,
      title: 'Real-time Analysis',
      description: 'WebSocket-powered real-time processing with live progress updates and interactive visualizations.',
    },
  ];

  const pipelineSteps = [
    {
      step: 1,
      title: 'Upload & Preprocess',
      description: 'Upload floor plan images and automatically segment rooms, walls, and doors using classical CV methods.',
    },
    {
      step: 2,
      title: 'Multi-Modal Encoding',
      description: 'Extract features using ViT for images, DistilBERT for text, and GNN for graph representations.',
    },
    {
      step: 3,
      title: 'Cross-Modal Fusion',
      description: 'Integrate all modalities using advanced transformer-based fusion with attention mechanisms.',
    },
    {
      step: 4,
      title: 'Task-Specific Analysis',
      description: 'Generate VQA answers, retrieve similar plans, and validate architectural constraints.',
    },
    {
      step: 5,
      title: 'Explainable Results',
      description: 'Provide comprehensive explanations with attention maps, constraint traces, and repair suggestions.',
    },
  ];

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center space-y-6">
        <div className="space-y-4">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
            Multi-Modal Floor Plan
            <span className="text-primary block">Understanding & Reasoning</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Research-grade AI system for comprehensive floor plan analysis using Vision Transformers, 
            Graph Neural Networks, and explainable AI techniques.
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button size="lg" asChild>
            <Link to="/analyze" className="flex items-center space-x-2">
              <Upload className="h-5 w-5" />
              <span>Start Analysis</span>
            </Link>
          </Button>
          <Button size="lg" variant="outline" asChild>
            <Link to="/docs" className="flex items-center space-x-2">
              <FileText className="h-5 w-5" />
              <span>View Documentation</span>
            </Link>
          </Button>
        </div>
      </section>

      {/* Features Section */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold">Advanced AI Capabilities</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Cutting-edge multi-modal AI technology for comprehensive floor plan understanding
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card key={index} className="h-full">
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Icon className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-base">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Pipeline Section */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold">End-to-End Pipeline</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Complete workflow from raw floor plan images to explainable AI insights
          </p>
        </div>
        
        <div className="space-y-6">
          {pipelineSteps.map((step, index) => (
            <Card key={index} className="relative">
              <CardContent className="p-6">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-lg">
                      {step.step}
                    </div>
                  </div>
                  <div className="flex-1 space-y-2">
                    <h3 className="text-xl font-semibold">{step.title}</h3>
                    <p className="text-muted-foreground">{step.description}</p>
                  </div>
                  {index < pipelineSteps.length - 1 && (
                    <div className="absolute left-6 top-16 w-0.5 h-8 bg-border"></div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Quick Actions */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold">Get Started</h2>
          <p className="text-lg text-muted-foreground">
            Choose your analysis type and start exploring floor plans with AI
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="group hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900">
                  <MessageSquare className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <CardTitle>VQA Analysis</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <CardDescription>
                Ask questions about floor plans and get intelligent answers with confidence scores.
              </CardDescription>
              <Button className="w-full group-hover:bg-primary/90" asChild>
                <Link to="/analyze?type=vqa" className="flex items-center justify-center space-x-2">
                  <span>Start VQA</span>
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>

          <Card className="group hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-green-100 dark:bg-green-900">
                  <Search className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
                <CardTitle>Similarity Search</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <CardDescription>
                Find similar floor plans using advanced embedding and contrastive learning.
              </CardDescription>
              <Button className="w-full group-hover:bg-primary/90" asChild>
                <Link to="/analyze?type=retrieval" className="flex items-center justify-center space-x-2">
                  <span>Find Similar</span>
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>

          <Card className="group hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-orange-100 dark:bg-orange-900">
                  <CheckCircle className="h-6 w-6 text-orange-600 dark:text-orange-400" />
                </div>
                <CardTitle>Validity Check</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <CardDescription>
                Validate floor plan structure and get repair suggestions with constraint tracing.
              </CardDescription>
              <Button className="w-full group-hover:bg-primary/90" asChild>
                <Link to="/analyze?type=validity" className="flex items-center justify-center space-x-2">
                  <span>Validate Plan</span>
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* CTA Section */}
      <section className="text-center space-y-6 py-12 bg-muted/50 rounded-lg">
        <h2 className="text-3xl font-bold">Ready to Analyze Floor Plans?</h2>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Upload your floor plan images and experience the power of multi-modal AI analysis
        </p>
        <Button size="lg" asChild>
          <Link to="/analyze" className="flex items-center space-x-2">
            <Upload className="h-5 w-5" />
            <span>Upload Floor Plan</span>
          </Link>
        </Button>
      </section>
    </div>
  );
};

export default HomePage;
