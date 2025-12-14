'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Mic, Volume2, VolumeX } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

export default function SamanthaChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  // Animated infinity symbol
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    let time = 0;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Create gradient for glow effect
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 150);
      gradient.addColorStop(0, 'rgba(255, 220, 200, 0.8)');
      gradient.addColorStop(0.5, 'rgba(255, 180, 150, 0.4)');
      gradient.addColorStop(1, 'rgba(255, 140, 100, 0)');

      // Draw outer glow
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Infinity symbol parameters
      const scale = isThinking ? 60 + Math.sin(time * 3) * 10 : 60;
      const lineWidth = isThinking ? 4 + Math.sin(time * 5) * 2 : 3;
      const pulseIntensity = isThinking ? 0.3 : 0.1;

      // Draw infinity symbol
      ctx.beginPath();
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.9 + Math.sin(time * 2) * pulseIntensity})`;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Parametric equations for infinity symbol (lemniscate)
      for (let t = 0; t <= Math.PI * 2; t += 0.01) {
        const x = centerX + (scale * Math.cos(t)) / (1 + Math.sin(t) * Math.sin(t));
        const y = centerY + (scale * Math.sin(t) * Math.cos(t)) / (1 + Math.sin(t) * Math.sin(t));
        
        if (t === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Add particles when thinking
      if (isThinking) {
        for (let i = 0; i < 3; i++) {
          const angle = time * 2 + (i * Math.PI * 2) / 3;
          const radius = 80 + Math.sin(time * 3 + i) * 20;
          const particleX = centerX + Math.cos(angle) * radius;
          const particleY = centerY + Math.sin(angle) * radius;
          
          ctx.beginPath();
          ctx.arc(particleX, particleY, 2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${0.6 + Math.sin(time * 4 + i) * 0.4})`;
          ctx.fill();
        }
      }

      time += isThinking ? 0.05 : 0.02;
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isThinking]);

  const sendMessage = async () => {
    if (!input.trim() || isProcessing) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsProcessing(true);
    setIsThinking(true);
    setStreamingMessage('');

    try {
      const socket = new WebSocket(`ws://localhost:8000/chat-stream`);
      
      socket.onopen = () => {
        const history = messages.map(m => [
          m.role === 'user' ? m.content : '', 
          m.role === 'assistant' ? m.content : ''
        ]).filter(([u, a]) => u || a);
        
        socket.send(JSON.stringify({
          message: userMessage,
          history: history.slice(-8),
          temperature: 0.82,
          max_tokens: 140
        }));
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.event === 'token') {
          setIsThinking(false);
          setStreamingMessage(prev => prev + data.data);
        } else if (data.event === 'done') {
          setMessages(prev => [...prev, { role: 'assistant', content: streamingMessage }]);
          setStreamingMessage('');
          setIsProcessing(false);
          setIsThinking(false);
          socket.close();
        } else if (data.event === 'error') {
          console.error('Stream error:', data.error);
          setIsProcessing(false);
          setIsThinking(false);
          socket.close();
        }
      };

      socket.onerror = () => {
        setIsProcessing(false);
        setIsThinking(false);
      };

    } catch (error) {
      console.error('Send error:', error);
      setIsProcessing(false);
      setIsThinking(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAudio(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setIsThinking(true);
    } catch (error) {
      console.error('Recording error:', error);
      alert('Microphone access denied. Please enable microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsThinking(false);
    }
  };

  const transcribeAudio = async (audioBlob) => {
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'audio.webm');

      const response = await fetch(`${API_BASE_URL}/stt`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('STT failed');

      const data = await response.json();
      if (data.text) {
        setInput(data.text);
      }
    } catch (error) {
      console.error('Transcription error:', error);
      alert('Failed to transcribe audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-900 via-orange-800 to-red-900 flex items-center justify-center p-4">
      <div className="w-full max-w-5xl h-[95vh] flex flex-col">
        
        {/* Main Display Area - HER OS Style */}
        <div className="flex-1 flex flex-col items-center justify-center relative">
          
          {/* Animated Infinity Symbol */}
          <div className="relative mb-8">
            <canvas 
              ref={canvasRef} 
              width="400" 
              height="300"
              className="opacity-90"
            />
            
            {/* Center text overlay */}
            {messages.length === 0 && !isProcessing && (
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-white/60 text-lg font-light tracking-wide">
                  Hello, I'm here
                </p>
              </div>
            )}
          </div>

          {/* Messages Display - Minimal HER Style */}
          <div className="w-full max-w-3xl px-8 mb-8">
            {messages.length > 0 && (
              <div className="space-y-6 max-h-[40vh] overflow-y-auto custom-scrollbar">
                {messages.slice(-4).map((msg, idx) => (
                  <div
                    key={idx}
                    className={`text-center animate-fade-in ${
                      msg.role === 'user' 
                        ? 'text-orange-200/80' 
                        : 'text-white/95'
                    }`}
                  >
                    <p className="text-lg font-light leading-relaxed tracking-wide">
                      {msg.role === 'user' ? '→ ' : ''}
                      {msg.content}
                    </p>
                  </div>
                ))}
                
                {streamingMessage && (
                  <div className="text-center animate-fade-in text-white/95">
                    <p className="text-lg font-light leading-relaxed tracking-wide">
                      {streamingMessage}
                      <span className="inline-block w-1 h-5 ml-2 bg-white/70 animate-pulse"></span>
                    </p>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Status Indicator */}
          {isProcessing && (
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
              <p className="text-white/50 text-sm font-light tracking-widest animate-pulse">
                {isThinking ? 'THINKING' : 'LISTENING'}
              </p>
            </div>
          )}
        </div>

        {/* Bottom Controls - Minimal */}
        <div className="px-8 pb-8">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center gap-4 mb-4">
              {/* Voice Button */}
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
                className={`p-4 rounded-full transition-all duration-500 ${
                  isRecording
                    ? 'bg-red-600/80 ring-4 ring-red-400/50 animate-pulse'
                    : 'bg-white/10 hover:bg-white/20'
                } backdrop-blur-sm disabled:opacity-30 disabled:cursor-not-allowed`}
              >
                <Mic className="text-white" size={20} />
              </button>

              {/* Text Input */}
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Say something..."
                disabled={isProcessing}
                className="flex-1 px-6 py-4 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 focus:border-white/40 focus:outline-none text-white placeholder-white/40 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
              />

              {/* Mute Toggle */}
              <button
                onClick={() => setIsMuted(!isMuted)}
                className="p-4 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-sm transition-all duration-300"
              >
                {isMuted ? 
                  <VolumeX className="text-white/70" size={20} /> : 
                  <Volume2 className="text-white/70" size={20} />
                }
              </button>
            </div>

            {/* Helper Text */}
            <p className="text-white/30 text-xs text-center font-light tracking-wider">
              Press Enter to send
            </p>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }

        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }

        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
        }

        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 2px;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }
      `}</style>
    </div>
  );
}