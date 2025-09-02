import React from 'react';
import { Wifi, WifiOff, AlertCircle } from 'lucide-react';
import { Badge } from './ui/badge';
import { useSocket } from '../contexts/SocketContext';

export const ConnectionStatus: React.FC = () => {
  const { isConnected, connectionStatus } = useSocket();

  const getStatusConfig = () => {
    switch (connectionStatus) {
      case 'connected':
        return {
          icon: Wifi,
          label: 'Connected',
          variant: 'default' as const,
          className: 'bg-green-500 hover:bg-green-600',
        };
      case 'error':
        return {
          icon: AlertCircle,
          label: 'Error',
          variant: 'destructive' as const,
          className: 'bg-red-500 hover:bg-red-600',
        };
      default:
        return {
          icon: WifiOff,
          label: 'Disconnected',
          variant: 'secondary' as const,
          className: 'bg-gray-500 hover:bg-gray-600',
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <Badge 
      variant={config.variant}
      className={`${config.className} text-white flex items-center space-x-1`}
    >
      <Icon className="h-3 w-3" />
      <span className="text-xs">{config.label}</span>
    </Badge>
  );
};
