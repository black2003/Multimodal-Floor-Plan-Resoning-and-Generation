import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Upload, X, Image as ImageIcon, AlertCircle } from 'lucide-react';
import { useToast } from '../hooks/use-toast';

interface ImageUploaderProps {
  onImageUpload: (imageData: string) => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageUpload }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const { toast } = useToast();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (PNG, JPG, JPEG, etc.)",
        variant: "destructive",
      });
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please upload an image smaller than 10MB",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setPreview(result);
      onImageUpload(result);
      setIsUploading(false);
      
      toast({
        title: "Image uploaded successfully",
        description: "Floor plan image is ready for analysis",
      });
    };

    reader.onerror = () => {
      setIsUploading(false);
      toast({
        title: "Upload failed",
        description: "Failed to read the image file",
        variant: "destructive",
      });
    };

    reader.readAsDataURL(file);
  }, [onImageUpload, toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    },
    multiple: false,
  });

  const clearImage = () => {
    setPreview(null);
    onImageUpload('');
  };

  if (preview) {
    return (
      <Card>
        <CardContent className="p-0">
          <div className="relative">
            <img
              src={preview}
              alt="Floor plan preview"
              className="w-full h-64 object-contain bg-muted rounded-t-lg"
            />
            <Button
              variant="destructive"
              size="sm"
              className="absolute top-2 right-2"
              onClick={clearImage}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="p-4">
            <div className="flex items-center space-x-2 text-sm text-muted-foreground">
              <ImageIcon className="h-4 w-4" />
              <span>Floor plan image uploaded successfully</span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-6">
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isDragActive 
              ? 'border-primary bg-primary/5' 
              : 'border-muted-foreground/25 hover:border-primary/50'
            }
            ${isUploading ? 'opacity-50 pointer-events-none' : ''}
          `}
        >
          <input {...getInputProps()} />
          
          <div className="space-y-4">
            <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center">
              {isUploading ? (
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
              ) : (
                <Upload className="h-6 w-6 text-muted-foreground" />
              )}
            </div>
            
            <div className="space-y-2">
              <h3 className="text-lg font-medium">
                {isUploading 
                  ? 'Uploading...' 
                  : isDragActive 
                    ? 'Drop the image here' 
                    : 'Upload Floor Plan Image'
                }
              </h3>
              <p className="text-sm text-muted-foreground">
                {isUploading 
                  ? 'Processing your image...'
                  : 'Drag and drop an image file here, or click to select'
                }
              </p>
            </div>
            
            {!isUploading && (
              <Button variant="outline" size="sm">
                Select Image
              </Button>
            )}
          </div>
        </div>
        
        <div className="mt-4 text-xs text-muted-foreground space-y-1">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-3 w-3" />
            <span>Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF</span>
          </div>
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-3 w-3" />
            <span>Maximum file size: 10MB</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
