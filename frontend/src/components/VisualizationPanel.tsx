import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  Image as ImageIcon, 
  Network, 
  Eye, 
  Download,
  Maximize2,
  Minimize2,
  RotateCcw
} from 'lucide-react';
import { AnalysisResult } from '../contexts/AnalysisContext';

interface VisualizationPanelProps {
  analysis: AnalysisResult;
}

export const VisualizationPanel: React.FC<VisualizationPanelProps> = ({ analysis }) => {
  const [activeTab, setActiveTab] = useState('original');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);

  const { metadata } = analysis;
  const { segmentation_info } = metadata;

  // Mock attention data for visualization
  const attentionData = {
    image: {
      patches: Array.from({ length: 196 }, (_, i) => ({
        id: i,
        attention: Math.random(),
        x: (i % 14) * 16,
        y: Math.floor(i / 14) * 16,
        width: 16,
        height: 16
      })),
      maxAttention: 1.0
    },
    graph: {
      nodes: Array.from({ length: segmentation_info.num_rooms + segmentation_info.num_doors }, (_, i) => ({
        id: i,
        type: i < segmentation_info.num_rooms ? 'room' : 'door',
        attention: Math.random(),
        x: Math.random() * 400 + 50,
        y: Math.random() * 300 + 50
      })),
      edges: Array.from({ length: segmentation_info.num_doors * 2 }, (_, i) => ({
        id: i,
        source: Math.floor(Math.random() * segmentation_info.num_rooms),
        target: Math.floor(Math.random() * segmentation_info.num_rooms),
        attention: Math.random()
      }))
    }
  };

  const drawAttentionOverlay = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw attention patches
    attentionData.image.patches.forEach(patch => {
      const alpha = (patch.attention / attentionData.image.maxAttention) * overlayOpacity;
      ctx.fillStyle = `rgba(255, 0, 0, ${alpha})`;
      ctx.fillRect(patch.x, patch.y, patch.width, patch.height);
    });
  };

  useEffect(() => {
    if (activeTab === 'attention' && canvasRef.current) {
      drawAttentionOverlay();
    }
  }, [activeTab, overlayOpacity]);

  const renderOriginalImage = () => (
    <div className="relative">
      <img
        src={analysis.image}
        alt="Floor plan"
        className="w-full h-auto rounded-lg border"
      />
    </div>
  );

  const renderAttentionVisualization = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium">Attention Overlay</span>
          <Badge variant="outline">Grad-CAM</Badge>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm">Opacity:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
            className="w-20"
          />
        </div>
      </div>
      
      <div className="relative">
        <img
          src={analysis.image}
          alt="Floor plan with attention"
          className="w-full h-auto rounded-lg border"
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          style={{ imageRendering: 'pixelated' }}
        />
      </div>
      
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="font-medium">Max Attention:</span>
          <span className="ml-2">{(attentionData.image.maxAttention * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="font-medium">Active Patches:</span>
          <span className="ml-2">{attentionData.image.patches.filter(p => p.attention > 0.5).length}</span>
        </div>
      </div>
    </div>
  );

  const renderGraphVisualization = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium">Graph Structure</span>
          <Badge variant="outline">GNN Attention</Badge>
        </div>
        <div className="flex items-center space-x-2">
          <Button size="sm" variant="outline">
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset Layout
          </Button>
        </div>
      </div>
      
      <div className="relative bg-muted rounded-lg p-4 min-h-[400px]">
        <svg width="100%" height="400" className="border rounded">
          {/* Render edges */}
          {attentionData.graph.edges.map(edge => {
            const sourceNode = attentionData.graph.nodes[edge.source];
            const targetNode = attentionData.graph.nodes[edge.target];
            if (!sourceNode || !targetNode) return null;
            
            return (
              <line
                key={edge.id}
                x1={sourceNode.x}
                y1={sourceNode.y}
                x2={targetNode.x}
                y2={targetNode.y}
                stroke={`rgba(59, 130, 246, ${edge.attention})`}
                strokeWidth={2 + edge.attention * 3}
                className="graph-edge"
              />
            );
          })}
          
          {/* Render nodes */}
          {attentionData.graph.nodes.map(node => (
            <g key={node.id}>
              <circle
                cx={node.x}
                cy={node.y}
                r={15 + node.attention * 10}
                fill={node.type === 'room' ? '#10b981' : '#f59e0b'}
                fillOpacity={0.7 + node.attention * 0.3}
                stroke={node.type === 'room' ? '#059669' : '#d97706'}
                strokeWidth={2}
                className="graph-node cursor-pointer"
              />
              <text
                x={node.x}
                y={node.y + 5}
                textAnchor="middle"
                className="text-xs font-medium fill-white"
              >
                {node.id}
              </text>
            </g>
          ))}
        </svg>
      </div>
      
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <span className="font-medium">Total Nodes:</span>
          <span className="ml-2">{attentionData.graph.nodes.length}</span>
        </div>
        <div>
          <span className="font-medium">Total Edges:</span>
          <span className="ml-2">{attentionData.graph.edges.length}</span>
        </div>
        <div>
          <span className="font-medium">Avg Attention:</span>
          <span className="ml-2">
            {(attentionData.graph.nodes.reduce((sum, n) => sum + n.attention, 0) / attentionData.graph.nodes.length * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );

  const renderSegmentationVisualization = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium">Segmentation Results</span>
          <Badge variant="outline">Classical CV</Badge>
        </div>
      </div>
      
      <div className="relative">
        <img
          src={analysis.image}
          alt="Floor plan segmentation"
          className="w-full h-auto rounded-lg border"
        />
        {/* Overlay segmentation results */}
        <div className="absolute top-4 left-4 bg-black/70 text-white p-2 rounded">
          <div className="text-xs space-y-1">
            <div>Rooms: {segmentation_info.num_rooms}</div>
            <div>Doors: {segmentation_info.num_doors}</div>
            <div>Walls: {segmentation_info.num_walls}</div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600">{segmentation_info.num_rooms}</div>
            <div className="text-sm text-muted-foreground">Rooms</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">{segmentation_info.num_doors}</div>
            <div className="text-sm text-muted-foreground">Doors</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-gray-600">{segmentation_info.num_walls}</div>
            <div className="text-sm text-muted-foreground">Walls</div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Eye className="h-5 w-5" />
              <span>Visualization</span>
            </CardTitle>
            <CardDescription>
              Interactive visualizations of analysis results
            </CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setIsFullscreen(!isFullscreen)}
            >
              {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
            <Button size="sm" variant="outline">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="original" className="flex items-center space-x-2">
              <ImageIcon className="h-4 w-4" />
              <span>Original</span>
            </TabsTrigger>
            <TabsTrigger value="attention" className="flex items-center space-x-2">
              <Eye className="h-4 w-4" />
              <span>Attention</span>
            </TabsTrigger>
            <TabsTrigger value="graph" className="flex items-center space-x-2">
              <Network className="h-4 w-4" />
              <span>Graph</span>
            </TabsTrigger>
            <TabsTrigger value="segmentation" className="flex items-center space-x-2">
              <ImageIcon className="h-4 w-4" />
              <span>Segmentation</span>
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="original" className="mt-4">
            {renderOriginalImage()}
          </TabsContent>
          
          <TabsContent value="attention" className="mt-4">
            {renderAttentionVisualization()}
          </TabsContent>
          
          <TabsContent value="graph" className="mt-4">
            {renderGraphVisualization()}
          </TabsContent>
          
          <TabsContent value="segmentation" className="mt-4">
            {renderSegmentationVisualization()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};
