import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from './ui/button';
import { ThemeToggle } from './ThemeToggle';
import { ConnectionStatus } from './ConnectionStatus';
import { 
  Home, 
  Search, 
  Brain, 
  BookOpen, 
  Menu,
  X
} from 'lucide-react';
import { useState } from 'react';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navigation = [
    { name: 'Home', href: '/', icon: Home },
    { name: 'Analyze', href: '/analyze', icon: Search },
    { name: 'Explain', href: '/explain', icon: Brain },
    { name: 'Docs', href: '/docs', icon: BookOpen },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          {/* Logo */}
          <div className="flex items-center space-x-2">
            <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
              <Brain className="h-5 w-5 text-primary-foreground" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-bold">Floor Plan AI</h1>
              <p className="text-xs text-muted-foreground">Multi-Modal Understanding</p>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Button
                  key={item.name}
                  variant={isActive(item.href) ? 'default' : 'ghost'}
                  size="sm"
                  asChild
                >
                  <Link to={item.href} className="flex items-center space-x-2">
                    <Icon className="h-4 w-4" />
                    <span>{item.name}</span>
                  </Link>
                </Button>
              );
            })}
          </nav>

          {/* Right side controls */}
          <div className="flex items-center space-x-2">
            <ConnectionStatus />
            <ThemeToggle />
            
            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              {isMobileMenuOpen ? (
                <X className="h-5 w-5" />
              ) : (
                <Menu className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden border-t bg-background">
            <nav className="container py-4 space-y-2">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <Button
                    key={item.name}
                    variant={isActive(item.href) ? 'default' : 'ghost'}
                    size="sm"
                    className="w-full justify-start"
                    asChild
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    <Link to={item.href} className="flex items-center space-x-2">
                      <Icon className="h-4 w-4" />
                      <span>{item.name}</span>
                    </Link>
                  </Button>
                );
              })}
            </nav>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="container py-6">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t bg-background">
        <div className="container py-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                Multi-Modal Floor Plan Understanding & Reasoning
              </span>
            </div>
            <div className="text-sm text-muted-foreground">
              © 2024 Floor Plan AI. Research-grade AI for architectural analysis.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
