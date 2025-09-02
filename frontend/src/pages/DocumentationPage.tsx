import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { 
  BookOpen, 
  Code, 
  Zap, 
  Brain, 
  FileText,
  Download,
  ExternalLink,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';

const DocumentationPage: React.FC = () => {
  const architectureComponents = [
    {
      name: 'Image Preprocessing',
      description: 'Normalize and resize images, classical segmentation for walls, rooms, and doors',
      technologies: ['OpenCV', 'PIL', 'scikit-image'],
      status: 'Implemented'
    },
    {
      name: 'Text Preprocessing',
      description: 'Parse captions, tokenize text, extract features for NLP models',
      technologies: ['NLTK', 'spaCy', 'Transformers'],
      status: 'Implemented'
    },
    {
      name: 'Graph Builder',
      description: 'Build graph representation with room nodes, adjacency edges, door connectivity',
      technologies: ['NetworkX', 'PyTorch Geometric'],
      status: 'Implemented'
    },
    {
      name: 'Vision Transformer',
      description: 'ViT encoder for image feature extraction with attention mechanisms',
      technologies: ['PyTorch', 'timm', 'Transformers'],
      status: 'Implemented'
    },
    {
      name: 'DistilBERT',
      description: 'Text encoder for natural language understanding and question processing',
      technologies: ['Transformers', 'PyTorch'],
      status: 'Implemented'
    },
    {
      name: 'Graph Neural Network',
      description: 'GNN encoder using Graph Attention Networks for structural understanding',
      technologies: ['PyTorch Geometric', 'GAT'],
      status: 'Implemented'
    },
    {
      name: 'Cross-Modal Fusion',
      description: 'Transformer-based fusion integrating all modality embeddings',
      technologies: ['PyTorch', 'Multi-Head Attention'],
      status: 'Implemented'
    },
    {
      name: 'VQA Head',
      description: 'Visual Question Answering with confidence estimation',
      technologies: ['PyTorch', 'Cross-Modal Attention'],
      status: 'Implemented'
    },
    {
      name: 'Retrieval Head',
      description: 'Contrastive learning for similarity search and plan retrieval',
      technologies: ['PyTorch', 'Contrastive Learning'],
      status: 'Implemented'
    },
    {
      name: 'Validity Head',
      description: 'Neuro-symbolic validation with constraint checking and repair suggestions',
      technologies: ['PyTorch', 'Rule-based Systems'],
      status: 'Implemented'
    },
    {
      name: 'Explainability',
      description: 'Grad-CAM, attention visualization, constraint tracing',
      technologies: ['Grad-CAM', 'Attention Maps', 'Rule Tracing'],
      status: 'Implemented'
    },
    {
      name: 'Frontend Interface',
      description: 'React-based interactive interface with real-time visualizations',
      technologies: ['React', 'TypeScript', 'TailwindCSS', 'WebSockets'],
      status: 'Implemented'
    }
  ];

  const apiEndpoints = [
    {
      method: 'POST',
      endpoint: '/api/upload',
      description: 'Upload floor plan image for analysis',
      parameters: ['image: File'],
      response: 'Analysis result with metadata'
    },
    {
      method: 'POST',
      endpoint: '/api/analyze',
      description: 'Analyze floor plan with text query',
      parameters: ['image: string', 'query: string', 'type: string'],
      response: 'Complete analysis results'
    },
    {
      method: 'POST',
      endpoint: '/api/vqa',
      description: 'Visual Question Answering',
      parameters: ['image: string', 'question: string'],
      response: 'Answer with confidence score'
    },
    {
      method: 'POST',
      endpoint: '/api/retrieve',
      description: 'Retrieve similar floor plans',
      parameters: ['image: string', 'top_k: number'],
      response: 'Similar plans with similarity scores'
    },
    {
      method: 'POST',
      endpoint: '/api/validate',
      description: 'Validate floor plan structure',
      parameters: ['image: string'],
      response: 'Validity score and repair suggestions'
    },
    {
      method: 'POST',
      endpoint: '/api/explain',
      description: 'Get explainability information',
      parameters: ['image: string', 'analysis_id: string', 'modality: string'],
      response: 'Detailed explanations and visualizations'
    }
  ];

  const renderOverview = () => (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">Multi-Modal Floor Plan Understanding</h2>
        <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
          A comprehensive AI research and application project for multi-modal floor-plan understanding and reasoning, 
          featuring deep learning modules, preprocessing, fusion, explainability, and a complete frontend interface.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>Multi-Modal AI</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Combines Vision Transformers, DistilBERT, and Graph Neural Networks for comprehensive floor plan analysis.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Real-time Processing</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              WebSocket-powered real-time analysis with live progress updates and interactive visualizations.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="h-5 w-5" />
              <span>Explainable AI</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Comprehensive explainability with Grad-CAM, attention visualization, and constraint tracing.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Key Features</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="font-medium">Core Capabilities</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Visual Question Answering (VQA)</li>
                <li>• Similarity retrieval and search</li>
                <li>• Structural validity checking</li>
                <li>• Multi-modal fusion and reasoning</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium">Technical Features</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Classical computer vision segmentation</li>
                <li>• Graph-based spatial reasoning</li>
                <li>• Cross-modal attention mechanisms</li>
                <li>• Neuro-symbolic validation</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderArchitecture = () => (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">System Architecture</h2>
        <p className="text-lg text-muted-foreground">
          Complete end-to-end pipeline from raw images to explainable AI insights
        </p>
      </div>

      <div className="space-y-4">
        {architectureComponents.map((component, index) => (
          <Card key={index}>
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="space-y-2">
                  <div className="flex items-center space-x-3">
                    <h3 className="text-lg font-semibold">{component.name}</h3>
                    <Badge 
                      variant={component.status === 'Implemented' ? 'default' : 'secondary'}
                      className="flex items-center space-x-1"
                    >
                      {component.status === 'Implemented' ? (
                        <CheckCircle className="h-3 w-3" />
                      ) : (
                        <AlertCircle className="h-3 w-3" />
                      )}
                      <span>{component.status}</span>
                    </Badge>
                  </div>
                  <p className="text-muted-foreground">{component.description}</p>
                  <div className="flex flex-wrap gap-1">
                    {component.technologies.map((tech, i) => (
                      <Badge key={i} variant="outline" className="text-xs">
                        {tech}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  const renderAPI = () => (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">API Documentation</h2>
        <p className="text-lg text-muted-foreground">
          RESTful API endpoints for floor plan analysis
        </p>
      </div>

      <div className="space-y-4">
        {apiEndpoints.map((endpoint, index) => (
          <Card key={index}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Badge 
                    variant={endpoint.method === 'POST' ? 'default' : 'secondary'}
                    className="font-mono"
                  >
                    {endpoint.method}
                  </Badge>
                  <code className="text-sm font-mono bg-muted px-2 py-1 rounded">
                    {endpoint.endpoint}
                  </code>
                </div>
              </div>
              <CardDescription>{endpoint.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <h4 className="text-sm font-medium mb-2">Parameters:</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  {endpoint.parameters.map((param, i) => (
                    <li key={i} className="font-mono">{param}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-medium mb-2">Response:</h4>
                <p className="text-sm text-muted-foreground">{endpoint.response}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>WebSocket Events</CardTitle>
          <CardDescription>
            Real-time communication for live analysis updates
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium mb-2">Client → Server</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li><code>analyze_floorplan</code> - Start analysis</li>
                <li><code>request_explanation</code> - Get explanations</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Server → Client</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li><code>analysis_progress</code> - Progress updates</li>
                <li><code>analysis_result</code> - Final results</li>
                <li><code>analysis_error</code> - Error messages</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderSetup = () => (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">Setup & Installation</h2>
        <p className="text-lg text-muted-foreground">
          Get started with the multi-modal floor plan understanding system
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Code className="h-5 w-5" />
              <span>Backend Setup</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div>
                <h4 className="font-medium">1. Install Dependencies</h4>
                <div className="bg-muted p-3 rounded-lg mt-2">
                  <code className="text-sm">pip install -r backend/requirements.txt</code>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium">2. Download Models</h4>
                <div className="bg-muted p-3 rounded-lg mt-2">
                  <code className="text-sm">python -m spacy download en_core_web_sm</code>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium">3. Start Server</h4>
                <div className="bg-muted p-3 rounded-lg mt-2">
                  <code className="text-sm">python backend/app/main.py</code>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Frontend Setup</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div>
                <h4 className="font-medium">1. Install Dependencies</h4>
                <div className="bg-muted p-3 rounded-lg mt-2">
                  <code className="text-sm">npm install</code>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium">2. Start Development Server</h4>
                <div className="bg-muted p-3 rounded-lg mt-2">
                  <code className="text-sm">npm start</code>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium">3. Build for Production</h4>
                <div className="bg-muted p-3 rounded-lg mt-2">
                  <code className="text-sm">npm run build</code>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System Requirements</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="font-medium">Backend Requirements</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Python 3.8+</li>
                <li>• PyTorch 2.0+</li>
                <li>• CUDA 11.8+ (optional, for GPU acceleration)</li>
                <li>• 8GB+ RAM recommended</li>
                <li>• 10GB+ disk space for models</li>
              </ul>
            </div>
            <div className="space-y-3">
              <h4 className="font-medium">Frontend Requirements</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Node.js 16+</li>
                <li>• Modern web browser</li>
                <li>• 2GB+ RAM</li>
                <li>• WebGL support (for visualizations)</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Quick Start Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4">
            <div className="flex items-start space-x-4">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                1
              </div>
              <div>
                <h4 className="font-medium">Clone Repository</h4>
                <div className="bg-muted p-2 rounded mt-1">
                  <code className="text-sm">git clone [repository-url]</code>
                </div>
              </div>
            </div>
            
            <div className="flex items-start space-x-4">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                2
              </div>
              <div>
                <h4 className="font-medium">Setup Backend</h4>
                <div className="bg-muted p-2 rounded mt-1">
                  <code className="text-sm">cd backend && pip install -r requirements.txt</code>
                </div>
              </div>
            </div>
            
            <div className="flex items-start space-x-4">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                3
              </div>
              <div>
                <h4 className="font-medium">Setup Frontend</h4>
                <div className="bg-muted p-2 rounded mt-1">
                  <code className="text-sm">cd frontend && npm install</code>
                </div>
              </div>
            </div>
            
            <div className="flex items-start space-x-4">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                4
              </div>
              <div>
                <h4 className="font-medium">Start Services</h4>
                <div className="bg-muted p-2 rounded mt-1">
                  <code className="text-sm"># Terminal 1: python backend/app/main.py</code><br/>
                  <code className="text-sm"># Terminal 2: npm start</code>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <h1 className="text-3xl font-bold">Documentation</h1>
        <p className="text-muted-foreground">
          Complete documentation for the Multi-Modal Floor Plan Understanding system
        </p>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="architecture">Architecture</TabsTrigger>
          <TabsTrigger value="api">API</TabsTrigger>
          <TabsTrigger value="setup">Setup</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="mt-6">
          {renderOverview()}
        </TabsContent>
        
        <TabsContent value="architecture" className="mt-6">
          {renderArchitecture()}
        </TabsContent>
        
        <TabsContent value="api" className="mt-6">
          {renderAPI()}
        </TabsContent>
        
        <TabsContent value="setup" className="mt-6">
          {renderSetup()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DocumentationPage;
