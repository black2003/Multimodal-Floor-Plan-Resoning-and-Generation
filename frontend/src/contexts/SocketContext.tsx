import React, { createContext, useContext, useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

type SocketContextType = {
  socket: Socket | null;
  isConnected: boolean;
  connectionStatus: string;
};

const SocketContext = createContext<SocketContextType>({
  socket: null,
  isConnected: false,
  connectionStatus: 'disconnected',
});

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
};

export const SocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  useEffect(() => {
    const newSocket = io('http://localhost:5000', {
      transports: ['websocket'],
      timeout: 20000,
    });

    newSocket.on('connect', () => {
      console.log('Connected to server');
      setIsConnected(true);
      setConnectionStatus('connected');
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
      setIsConnected(false);
      setConnectionStatus('disconnected');
    });

    newSocket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      setConnectionStatus('error');
    });

    newSocket.on('status', (data) => {
      console.log('Server status:', data);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  return (
    <SocketContext.Provider value={{ socket, isConnected, connectionStatus }}>
      {children}
    </SocketContext.Provider>
  );
};
