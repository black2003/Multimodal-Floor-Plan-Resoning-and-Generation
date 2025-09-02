import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { 
  Brain, 
  Eye, 
  Network, 
  FileText, 
  BarChart3,
  Lightbulb,
  Target,
  Zap
} from 'lucide-react';

const ExplainabilityPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const explainabilityFeatures = [
    {
      icon: Eye,
      title: 'Image Attention',
      description: 'Grad-CAM and attention overlays showing which parts of the floor plan the AI focuses on',
      modality: 'Visual',
      examples: ['Room detection', 'Door placement', 'Spatial relationships']
    },
    {
      icon: FileText,
      title: 'Text Attention',
      description: 'Token-level attention highlighting important words in questions and captions',
      modality: 'Linguistic',
      examples: ['Question understanding', 'Keyword extraction', 'Semantic analysis']
    },
    {
      icon: Network,
      title: 'Graph Attention',
      description: 'Node and edge attention weights showing important relationships in the floor plan graph',
      modality: 'Structural',
      examples: ['Room connectivity', 'Door connections', 'Spatial hierarchy']
    },
    {
      icon: Target,
      title: 'Constraint Tracing',
      description: 'Rule-based validation with detailed constraint checking and violation explanations',
      modality: 'Symbolic',
      examples: ['Accessibility rules', 'Building codes', 'Design principles']
    }
  ];

  const explainabilityMethods = [
    {
      name: 'Grad-CAM',
      description: 'Gradient-weighted Class Activation Mapping for visual attention',
      type: 'Image',
      implementation: 'Vision Transformer attention layers'
    },
    {
      name: 'Attention Visualization',
      description: 'Multi-head attention weights across all modalities',
      type: 'Cross-Modal',
      implementation: 'Transformer attention matrices'
    },
    {
      name: 'Graph Attention',
      description: 'Graph Neural Network attention on nodes and edges',
      type: 'Graph',
      implementation: 'Graph Attention Networks (GAT)'
    },
    {
      name: 'Constraint Tracing',
      description: 'Neuro-symbolic rule execution and violation tracking',
      type: 'Symbolic',
      implementation: 'Rule-based constraint checker'
    },
    {
      name: 'Feature Attribution',
      description: 'Contribution of individual features to final predictions',
      type: 'Feature',
      implementation: 'Integrated gradients and SHAP'
    },
    {
      name: 'Counterfactual Analysis',
      description: 'What-if scenarios showing how changes affect predictions',
      type: 'Causal',
      implementation: 'Counterfactual generation'
    }
  ];

  const renderOverview = () => (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">Explainable AI for Floor Plans</h2>
        <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
          Comprehensive explainability across all modalities - image, text, graph, and symbolic reasoning
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {explainabilityFeatures.map((feature, index) => {
          const Icon = feature.icon;
          return (
            <Card key={index} className="h-full">
              <CardHeader>
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                    <Badge variant="outline" className="mt-1">{feature.modality}</Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <CardDescription className="text-base">
                  {feature.description}
                </CardDescription>
                <div>
                  <p className="text-sm font-medium mb-2">Examples:</p>
                  <div className="flex flex-wrap gap-1">
                    {feature.examples.map((example, i) => (
                      <Badge key={i} variant="secondary" className="text-xs">
                        {example}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );

  const renderMethods = () => (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">Explainability Methods</h2>
        <p className="text-lg text-muted-foreground">
          Advanced techniques for understanding AI decision-making
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {explainabilityMethods.map((method, index) => (
          <Card key={index}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{method.name}</CardTitle>
                <Badge variant="outline">{method.type}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <CardDescription className="text-base">
                {method.description}
              </CardDescription>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-sm font-medium">Implementation:</p>
                <p className="text-sm text-muted-foreground">{method.implementation}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  const renderVisualization = () => (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">Interactive Visualizations</h2>
        <p className="text-lg text-muted-foreground">
          Explore attention patterns and explanations interactively
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Eye className="h-5 w-5" />
              <span>Image Attention</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
              <div className="text-center space-y-2">
                <Eye className="h-12 w-12 mx-auto text-muted-foreground" />
                <p className="text-muted-foreground">Grad-CAM Visualization</p>
                <p className="text-sm text-muted-foreground">Upload an image to see attention maps</p>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Features:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Real-time attention overlay</li>
                <li>• Adjustable opacity controls</li>
                <li>• Patch-level attention scores</li>
                <li>• Export high-resolution maps</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Network className="h-5 w-5" />
              <span>Graph Attention</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
              <div className="text-center space-y-2">
                <Network className="h-12 w-12 mx-auto text-muted-foreground" />
                <p className="text-muted-foreground">Graph Visualization</p>
                <p className="text-sm text-muted-foreground">Interactive node and edge attention</p>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Features:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Interactive graph exploration</li>
                <li>• Node attention highlighting</li>
                <li>• Edge weight visualization</li>
                <li>• Layout optimization</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="h-5 w-5" />
              <span>Text Attention</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
              <div className="text-center space-y-2">
                <FileText className="h-12 w-12 mx-auto text-muted-foreground" />
                <p className="text-muted-foreground">Token Attention</p>
                <p className="text-sm text-muted-foreground">Word-level attention weights</p>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Features:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Token-level highlighting</li>
                <li>• Attention matrix visualization</li>
                <li>• Layer-wise attention comparison</li>
                <li>• Semantic role analysis</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="h-5 w-5" />
              <span>Constraint Tracing</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
              <div className="text-center space-y-2">
                <Target className="h-12 w-12 mx-auto text-muted-foreground" />
                <p className="text-muted-foreground">Rule Execution</p>
                <p className="text-sm text-muted-foreground">Constraint violation tracking</p>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Features:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Rule execution trace</li>
                <li>• Violation highlighting</li>
                <li>• Repair suggestions</li>
                <li>• Constraint dependency graph</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  const renderUsage = () => (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">How to Use Explainability</h2>
        <p className="text-lg text-muted-foreground">
          Step-by-step guide to understanding AI explanations
        </p>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Quick Start</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-start space-x-4">
                <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                  1
                </div>
                <div>
                  <h4 className="font-medium">Upload Floor Plan</h4>
                  <p className="text-sm text-muted-foreground">
                    Upload a floor plan image to the analysis page
                  </p>
                </div>
              </div>
              
              <div className="flex items-start space-x-4">
                <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                  2
                </div>
                <div>
                  <h4 className="font-medium">Run Analysis</h4>
                  <p className="text-sm text-muted-foreground">
                    Choose analysis type and start the AI processing
                  </p>
                </div>
              </div>
              
              <div className="flex items-start space-x-4">
                <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                  3
                </div>
                <div>
                  <h4 className="font-medium">View Explanations</h4>
                  <p className="text-sm text-muted-foreground">
                    Navigate to the Explainability tab to see detailed explanations
                  </p>
                </div>
              </div>
              
              <div className="flex items-start space-x-4">
                <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                  4
                </div>
                <div>
                  <h4 className="font-medium">Explore Visualizations</h4>
                  <p className="text-sm text-muted-foreground">
                    Interact with attention maps, graph visualizations, and constraint traces
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Lightbulb className="h-5 w-5" />
              <span>Best Practices</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium">For VQA Analysis</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Check text attention for question understanding</li>
                  <li>• Examine image attention for relevant regions</li>
                  <li>• Compare cross-modal attention patterns</li>
                </ul>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium">For Validity Checking</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Review constraint violation traces</li>
                  <li>• Examine graph attention for connectivity</li>
                  <li>• Check repair suggestions and priorities</li>
                </ul>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium">For Similarity Search</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Analyze feature importance scores</li>
                  <li>• Compare attention patterns across plans</li>
                  <li>• Examine embedding space visualizations</li>
                </ul>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium">General Tips</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Use multiple explanation methods together</li>
                  <li>• Compare attention across different layers</li>
                  <li>• Export visualizations for documentation</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <h1 className="text-3xl font-bold">Explainability</h1>
        <p className="text-muted-foreground">
          Understand how the AI analyzes floor plans with comprehensive explainability features
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="methods">Methods</TabsTrigger>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="usage">Usage Guide</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="mt-6">
          {renderOverview()}
        </TabsContent>
        
        <TabsContent value="methods" className="mt-6">
          {renderMethods()}
        </TabsContent>
        
        <TabsContent value="visualization" className="mt-6">
          {renderVisualization()}
        </TabsContent>
        
        <TabsContent value="usage" className="mt-6">
          {renderUsage()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ExplainabilityPage;
