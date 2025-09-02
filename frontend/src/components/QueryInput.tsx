import React from 'react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { MessageSquare, Send } from 'lucide-react';

interface QueryInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  onSubmit?: () => void;
}

export const QueryInput: React.FC<QueryInputProps> = ({ 
  value, 
  onChange, 
  placeholder = "Ask a question about the floor plan...",
  onSubmit 
}) => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim() && onSubmit) {
      onSubmit();
    }
  };

  const exampleQuestions = [
    "How many bedrooms are in this floor plan?",
    "Where is the kitchen located?",
    "What is the total area of the living room?",
    "Are there any bathrooms on the second floor?",
    "How many doors lead to the outside?",
  ];

  return (
    <Card>
      <CardContent className="p-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center space-x-2">
            <MessageSquare className="h-5 w-5 text-muted-foreground" />
            <span className="text-sm font-medium">Your Question</span>
          </div>
          
          <div className="space-y-2">
            <textarea
              value={value}
              onChange={(e) => onChange(e.target.value)}
              placeholder={placeholder}
              className="w-full min-h-[100px] p-3 border border-input rounded-md bg-background text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
              rows={3}
            />
            
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">
                {value.length} characters
              </span>
              
              {onSubmit && (
                <Button 
                  type="submit" 
                  size="sm" 
                  disabled={!value.trim()}
                  className="flex items-center space-x-2"
                >
                  <Send className="h-4 w-4" />
                  <span>Submit</span>
                </Button>
              )}
            </div>
          </div>
        </form>
        
        {/* Example Questions */}
        <div className="mt-4 space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Example questions:</p>
          <div className="space-y-1">
            {exampleQuestions.slice(0, 3).map((question, index) => (
              <button
                key={index}
                onClick={() => onChange(question)}
                className="block w-full text-left text-xs text-muted-foreground hover:text-foreground transition-colors p-2 rounded hover:bg-muted"
              >
                "{question}"
              </button>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
